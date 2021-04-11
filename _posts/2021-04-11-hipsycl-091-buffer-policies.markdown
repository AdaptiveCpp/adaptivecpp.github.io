---
layout: single
title:  "hipSYCL 0.9.1 features: Asynchronous buffers and explicit buffer policies"
date:   2021-04-11 19:28:06 +0100
categories: hipsycl extension
---

This post is part of a series where we discuss some features of the brandnew hipSYCL 0.9.1. Today I want to take a closer look at

# Asynchronous buffers and explicit buffer policies

This is a new extension in hipSYCL that can make code using `sycl::buffer` objects **much clearer while also improving performance**. Interested? Then this blog post is for you.

## Motivation 1: Buffers are complicated

A `sycl::buffer` is a very complicated object. Depending on a combination of multiple factors the semantics of a `sycl::buffer` can be very different. Will it operate directly on input pointers or will it copy input data to some internal storage? Will it submit a writeback in the destructor to copy data back to host?

I have frequently noticed users getting this wrong. This can either lead to correctness issues, for example
* the buffer operates directly on the input pointer, while the user has only intended to provide it as a source of initial data and wanted to reuse it after buffer construction
* no writeback is issued even though the user expected data to be copied back to host.

Or performance bugs might be introduced - these are arguably even worse because you might not notice them right away and they might be difficult to find. Some performance bugs that I have seen in user code are:
* The buffer issued an unexpected writeback, and thus copied data back to host without the user intending it
* The buffer did not operate directly on the pointer provided in the constructor, but instead first copied the data to internal storage which broke performance assumptions on the CPU backend.

## Motivation 2: The buffer destructor antipattern

In addition, there is a related performance antipattern that I have noticed frequently. Consider the following code:


{% highlight cpp %}

T* ptr1 = ...;
T* ptr2 = ...;
T* ptr3 = ...;
sycl::range<1> size = ...;

{
  sycl::buffer<T> b1{ptr1, size};
  sycl::buffer<T> b2{ptr2, size};
  sycl::buffer<T> b3{ptr3, size};

  // Kernels using b1, b2, b3

} // Destructors issue write-back

{% endhighlight %}

We construct three buffers that get an input pointer and then, when the scope closes, issue a writeback in their destructors. The problem here is that the execution of writebacks is not really efficient. The SYCL specification requires that in the destructor, a `buffer` has to wait for the completion of all operations that use it. This means that the following sequence of operations will be executed:

1. `b3.~buffer()` runs: submit writeback, wait for completion
2. `b2.~buffer()` runs: submit writeback, wait for completion
3. `b1.~buffer()` runs: submit writeback, wait for completion

Here we have multiple unnecessary cases of synchronization. For performance it is always better to submit all available work asynchronously, and then wait as late as possible with as few wait calls as possible. So, something like the following will in general perform better:

1. `b3.~buffer()` runs: submit writeback asynchronously
2. `b2.~buffer()` runs: submit writeback asynchronously
3. `b1.~buffer()` runs: submit writeback asynchronously
4. Maybe do some other work while the writebacks are being processed
5. Wait for all writebacks to complete

This has multiple advantages:
1. The SYCL implementation can process a larger task graph consisting of multiple writebacks as well as any other operations that might have been submitted previously, allowing for more optimization opportunities
2. There is less latency between the writebacks when they are processed by the SYCL backend and hardware, because there is no synchronization in between them.
3. The execution of writeback can be overlapped with other work on the host if the wait is executed later.

*Note:* While the worst case is clearly when the buffers submit writebacks as in this example, even if the buffers do not submit a writeback, there might still be a negative performance impact: Because all buffer destructors need to wait individually, they cause individual and potentially unnecessary flushes of the SYCL task graph.

## Enter explicit buffer policies

To address both the destructor antipattern as well as the complexity of buffers, hipSYCL 0.9.1 introduces *explicit buffer policies*, which allow the user to explicitly specify the desired behavior of a buffer. We introduce the following terminology:

| | Destructor blocks? | Writes back ? | Uses external storage? |
| ----- | ----- | ----- | ----- |
| yes | `sync_` | `_writeback_` | `view` |
| no  | `async_` | - | `buffer` |

For example, a `sync_writeback_view` refers to the behavior where the destructor blocks (`sync`), a writeback will be issued in the destructor (`writeback`)  and the buffer will operate directly on provided input data pointers (`view`).

These behaviors are not expressed as new C++ types, but as regular `sycl::buffer` objects that are initialized with special buffer properties. buffers with explicit behaviors are constructed using factory functions such as  `buffer<T, Dim> make_sync_buffer(...)`.
Since these functions still return a `sycl::buffer<T, Dim>`, explicit buffer behaviors integrate well with existing SYCL code that relies on the `sycl::buffer` type.

Using those factory functions instead of directly constructing `sycl::buffer` objects significantly improves code clarity - the programmer can now see with one quick glance at the function call what is going to happen, and what performance implications there are.

### View

Buffers of `view` behavior operate directly on the provided input pointer when running on the CPU backend. The pointer must be considered as being in use by the buffer until all operations that the buffer is involved in have completed, including potential writebacks.

### Buffer

Buffers of `buffer` behavior will not operate directly on optionally provided input pointers. If an input data pointer is provided, the data content will be copied to internal storage. The pointer is safe to use (or delete) as desired by the user after the buffer constructor returns.

### Writeback

Buffers of `writeback` behavior will submit a writeback operation to migrate data back to host in the destructor. This will only lead to an actual data copy if the data on the host is outdated. With hipSYCL explicit buffer behaviors, a writeback needs to be explicitly requested by invoking a buffer factory function with `writeback` in its name. This prevents users accidentally introducing performance bugs by means of unnecessary writebacks.

### sync/async

Only buffers with `sync` behavior block in their destructor. Buffers of `async` behavior do not - and therefore can be used to solve the buffer destructor performance antipattern:

{% highlight cpp %}

sycl::queue q;
{
  auto b1 = sycl::make_async_writeback_view(ptr1, size, q);
  auto b2 = sycl::make_async_writeback_view(ptr2, size, q);
  auto b3 = sycl::make_async_writeback_view(ptr3, size, q);
  
  // Submit kernels operating on b1,b2,b3 here
} // Non-blocking buffer destructors

// At some later point, use q.wait() to wait
// for all writebacks
q.wait();
{% endhighlight %}

Here async writeback views are used that do not block in their destructor. hipSYCL guarantees that memory allocated by buffer objects will not be freed if there are still operations in flight utilizing those allocations, so kernels and other operations using the buffer objects will complete successfully even if the user-facing buffer object has already been destroyed.

**For performance it should be considered best practice to use the async behaviors by default and only use the sync variants when it is absolutely necessary.**

## API reference

Not every combination of buffer behaviors makes sense. hipSYCL currently supports the following factory functions:

{% highlight cpp %}

/// Only uses internal storage, 
/// no writeback, 
/// blocking destructor
template <class T, int Dim>
buffer<T, Dim> make_sync_buffer(
    sycl::range<Dim> r);

/// Only uses internal storage,
/// no writeback,
/// blocking destructor.
/// Data pointed to by ptr is copied to internal storage.
template <class T, int Dim>
buffer<T, Dim> make_sync_buffer(
    const T* ptr, sycl::range<Dim> r);

/// Only internal storage, 
/// no writeback,
/// non-blocking destructor
template <class T, int Dim>
buffer<T, Dim> make_async_buffer(
    sycl::range<Dim> r);

/// Only internal storage,
/// no writeback,
/// non-blocking destructor.
/// Data pointed to by ptr is copied to internal storage.
template <class T, int Dim>
buffer<T, Dim> make_async_buffer(
    const T* ptr, sycl::range<Dim> r);

/// Uses provided storage,
/// writes back,
/// blocking destructor.
/// Directly operates on host_view_ptr.
template <class T, int Dim>
buffer<T, Dim> make_sync_writeback_view(
    T* host_view_ptr, sycl::range<Dim> r);

/// Uses provided storage,
/// writes back,
/// non-blocking destructor.
/// Directly operates on host_view_ptr.
/// The provided queue can be used by the user to 
/// wait for the writeback to complete.
template <class T, int Dim>
buffer<T, Dim> make_async_writeback_view(
    T* host_view_ptr, sycl::range<Dim> r,
    const sycl::queue& q);

/// Uses provided storage,
/// does not write back,
/// blocking destructor.
/// Directly operates on host_view_ptr.
template <class T, int Dim>
buffer<T, Dim> make_sync_view(
    T* host_view_ptr, sycl::range<Dim> r);

/// Uses provided storage,
/// does not write back,
/// non-blocking destructor.
/// Directly operates on host_view_ptr.
template <class T, int Dim>
buffer<T, Dim> make_async_view(
    T* host_view_ptr, sycl::range<Dim> r);

/// Additional factory functions exist for 
/// buffer-USM interoperability.
/// Those will be covered in more detail in a future blog post.

{% endhighlight %}

For the full API reference, see the [hipSYCL documentation](https://github.com/illuhad/hipSYCL/blob/develop/doc/explicit-buffer-policies.md).


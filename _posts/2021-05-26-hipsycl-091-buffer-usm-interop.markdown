---
layout: single
title:  "hipSYCL 0.9.1 features: buffer-USM interoperability"
date:   2021-05-26 18:30:00 +0100
categories: hipsycl extension
---

This post is part of a series where we discuss some features of hipSYCL 0.9.1. Today's topic is interoperability between buffers and USM pointers.

# Why it matters

SYCL 2020 features two major memory management models, both of which are supported by hipSYCL:
1. The traditional buffer-accessor model that has already been available in the old SYCL 1.2.1. In this model, a task graph is constructed automatically based on access conflicts between access specifications described by `accessor` objects. These `accessor` objects are also used to access data in kernels. The buffer-accessor model provides the SYCL runtime with a lot of information about how much data is used and how it is used. This can help scheduling, and enables automatic optimizations such as overlap of data transfers and kernels.
2. The pointer-based USM model that was introduced in SYCL 2020. Here, allocations are managed explicitly and (unless shared allocations are used) data must be copied explicitly between host and device. The USM model provides more control to the user compared to the buffer-USM model, at the cost of requiring the user to do work that the runtime can do automatically in the buffer-accessor model. It also forces the programmer to think in a model of a host-device dichotomy, which may not be an ideal fit when CPUs are targeted. On the other hand, it is usually considerably easier to port existing pointer-based code to SYCL using the USM model compared to the buffer-accessor model.

It is apparent that both models have valid use cases and are complementary. However, in SYCL 2020, there is barely any interoperability between the two. Accessing data that is stored in a buffer using a USM pointer requires launching a custom kernel that explicitly copies all data elements from the buffer into a USM pointer. This is both cumbersome and comes at a performance cost.

Consequently, once a codebase has started using one particular model, it is effectively locked into it. This is problematic for several reasons:

1. As the SYCL software ecosystem grows, there is a **real danger of ecosystem bifurcation** if no mechanisms are provided to cross from USM-land to buffer-land and vice versa. A SYCL library with a USM pointer API will be of little use for a SYCL application that is written using buffers and accessors.
2. SYCL is all about taking control when you want it, and letting SYCL do what it thinks is best otherwise. This allows to combine the best of two worlds: Low-level kernel optimizations for critical code paths, and the convenience of a high-level programming model for the remaining program. Consequently, **it should be possible to use USM pointers whenever we want detailed low-level control, and move to a more high-level model for other parts of the program**. Not having interoperability between them **can block potential incremental optimization paths during software development**.
3. Which model will be better in terms of performance or clarity is not always apparent, and might be different for different parts of the program. As outlined above, both have strengths and weaknesses, and are complementary. **We should therefore be able to mix buffers and USM pointers.**

# buffer-USM interoperability

To address these issues, hipSYCL 0.9.1 has introduced a comprehensive API for interoperability between USM pointers and buffers. In hipSYCL, you can always construct a buffer on top of existing USM pointers, or extract a USM pointer from a buffer -- completely without additional data copies.

hipSYCL is the first SYCL implementation to expose such a feature, and the reason is found easily: Buffer-USM interoperability in a meaningful, convenient and efficient way requires guarantees about the internal buffer behavior and SYCL implementation design that far exceed anything the SYCL specification guarantees.

We have therefore introduced an additional [hipSYCL runtime specification](https://github.com/illuhad/hipSYCL/blob/develop/doc/runtime-spec.md) that more rigorously defines buffer behavior. In particular hipSYCL makes the following guarantees that are crucial for buffer-USM interoperability:
* Buffers use USM pointers internally. All allocations a buffer performs are USM allocations, and buffers are entirely implemented on top of USM pointers.
* Allocations are persistent. Buffers guarantee that allocations, once they have been made, will remain valid at least until the end of buffer lifetime. Buffers will manage exactly one allocation per (physical) device.
* Buffers allocate lazily. When the buffer is used for the first time on a particular device, it will allocate memory large enough for all of the data such that no reallocations are needed for the lifetime of the buffer.

There are two cases to distinguish for buffer-USM interoperability:
1. Temporal composition: Here we just move memory allocations from USM pointers into a buffer or vice versa; at each point in time only either a USM pointer or a buffer exists for a given allocation.
2. The more complex case: Simultaneously accessing the same allocation as USM pointer and buffer. This is more complicated as it requires some correctness considerations by the programmer.

## Temporal composition

Let's focus on the simple case first: Assume we only want to turn an existing buffer into a USM pointer (or vice versa), but don't want to use them simultaneously. hipSYCL has a fairly intuitive API for that: `buffer::get_pointer()` to extract USM pointers and a special buffer constructor that accepts USM pointers.

{% highlight cpp %}

sycl::queue q;
std::size_t s = 1024;
int* mem = sycl::malloc_device<int>(s, q);

// Use mem as USM pointer
q.parallel_for(sycl::range{s}, 
    [=](sycl::id<1> idx){ mem[idx[0]] = idx[0]; });
// Make sure that USM operations terminate before
// using mem as buffer
q.wait();

// Construct buffer on top of existing USM pointer
{
  sycl::device dev = q.get_device();
  // Use mem for all operations for device dev. view() assumes
  // that the pointer holds valid data. If it should be considered empty,
  // use empty_view() instead.
  // Note the {} around the view: This is because we are actually passing
  // an std::vector. You can feed multiple USM pointers (one for each device)
  // into a buffer! Here, we only use one device.
  sycl::buffer<int> buff{
    {sycl::buffer_allocation::view(mem, dev)}, sycl::range{s}};
  
  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc{buff, cgh};
    cgh.parallel_for(sycl::range{s}, [=](sycl::id<1> idx){
      acc[idx] += 1;
    });
  });
  
  // Turn buffer into USM pointer again.
  // Note: get_pointer() returns nullptr if no allocation is available on a device,
  // e.g. if a buffer hasn't yet been used on a device (remember: lazy allocation!) 
  // or was not initialized with an appropriate view() object.
  // In this example, we know that the buffer has an allocation for this
  // device because we have given one in the constructor.
  int* mem_extracted = buff.get_pointer(dev);
  assert(mem_extracted == mem);
  
  // This makes sure that the buffer won't delete the allocation when
  // it goes out of scope, so we can use it afterwards.
  // By default, view() is non-owning, so in this example it's
  // not strictly necessary.
  buff.disown_allocation(dev);
} // Closing scope synchronizes all tasks operating on the buffer.

// Use USM pointer again
q.parallel_for(sycl::range{s}, ...).wait();

sycl::free(mem, q);
{% endhighlight %}

## Simultaneous USM pointers and buffers

If we want to have both USM pointers and buffers accessing the same allocation simultaneously, things get more complicated. In this scenario, it is crucial to understand that
1. Buffers automatically calculate dependencies to other operations by detecting conflicting accessors. If operations use the same allocation but without going through accessors, buffers cannot know about these additional dependencies -- the programmer must insert them manually.
2. Buffers automatically calculate necessary data transfers by tracking whether data is valid or outdated on a particular device. If data is modified through USM pointers without the buffer knowing of it, the internal data tracking of the buffer is off and no longer reflects reality. This can cause the buffer to emit data transfers that shouldn't take place, or omit data transfers when they might actually be required. To avoid this, we need to manually update the buffer's data tracking.

Here's an example that shows how it's done.
{% highlight cpp %}

sycl::queue q;
// Queue on a different device for later use
sycl::device other_dev = ...;
sycl::queue q2{other_dev};

std::size_t s = 1024;
sycl::buffer<int> buff{sycl::range{s}};

// Extract USM pointer - at this point we are not yet guaranteed
// that an allocation exists because memory is allocated lazily.
// We can however force preallocation of memory using the hipSYCL 
// handler::update extension (Not yet in hipSYCL 0.9.1, but in 
// current develop branch on github).
q.submit([&](sycl::handler& cgh){
  sycl::accessor acc{buff, cgh};
  cgh.update(acc);
});
// Also preallocate on another device for later use.
q2.submit([&](sycl::handler& cgh){
  sycl::accessor acc{buff, cgh};
  cgh.update(acc);
});
q.wait(); q2.wait();

// Since memory has now been allocated by the buffer, we can now extract
// an USM pointer.
int* usm_ptr = buff.get_pointer(q.get_device());

// Submit a kernel operating on buff
sycl::event evt = q.submit([&](sycl::handler& cgh){
  sycl::accessor acc{buff, cgh};
  cgh.parallel_for(sycl::range{s}, [=](sycl::id<1> idx){
    // Use acc here
  });
});
// Submit a USM kernel
sycl::event evt2 = q.submit([&](sycl::handler& cgh){
  // Important: Add dependency to the other kernel!
  cgh.depends_on(evt);
  cgh.parallel_for(sycl::range{s}, [=](sycl::id<1> idx){
    // Use usm_ptr here
  });
});
{% endhighlight %}
So far no surprises -- we just had to insert dependencies manually as expected. Let's now look at submitting work to a different device. When submitting USM operations to another device, we need to inform the buffer that there are writes taking place on that device, and that it should consider allocations on other devices as outdated after this point. We again use `handler::update()` for this.
{% highlight cpp %}

// This is necessary to allow the buffer to infer necessary data transfers correctly.
sycl::event evt3 = q2.submit([&](sycl::handler& cgh){
  // Depend on previous USM operation
  cgh.depends_on(evt2);
  // This is a read-write accessor - it's important that there's
  // a write in the access mode if we want to write to usm_ptr
  // in the next kernel.
  sycl::accessor acc{buff, cgh};
  cgh.update(acc);
})
int* usm_ptr2 = buff.get_pointer(q2.get_device());
sycl::event evt4 = q2.submit([&](sycl::handler& cgh){
  cgh.depends_on(evt3);
  cgh.parallel_for(sycl::range{s}, [=](sycl::id<1> idx){
    // Use usm_ptr2 here
  });
});
// End with operation on first device
q.submit([&](sycl::handler& cgh){
  // Buffer cannot know that USM kernel operates on same data,
  // so we need to manually insert a dependency.
  cgh.depends_on(evt4);
  // This accessor will trigger data migration back to
  // the first device because we are submitting to q
  // instead of q2
  sycl::accessor acc{buff, cgh};
  cgh.parallel_for(sycl::range{s}, [=](sycl::id<1> idx){
    // Use acc here
  });
});
{% endhighlight %}

In summary, even using buffers and USM pointers simultaneously for the same data is possible, but requires a solid understanding of SYCL and the guarantees that hipSYCL makes specifically.

Remember that buffers cannot know about USM kernels that utilize the same allocations, so always, always make sure to insert correct dependencies. Also, make sure to inform the buffer that an allocation has been *modified* so that it can correctly emit data transfers when an accessor is used for the buffer on a different device (including the host device). This can be done by constructing a  accessor with a suitable access mode -- either by using `handler::update()`, or by submitting a kernel that uses accessors.

In practice, this might be much simpler. If you are not working with complex task graphs, you could just use a SYCL 2020 in-order queue to avoid having to insert all those dependencies manually. And if you are only working on a single device, your `handler::update()` calls might not be required anymore.


## API reference

For the full API reference, see the [hipSYCL documentation](https://github.com/illuhad/hipSYCL/blob/develop/doc/buffer-usm-interop.md).


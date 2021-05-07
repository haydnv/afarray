# afarray
Convenience methods for working with ArrayFire arrays in Rust, compatible with
[number-general](http://github.com/haydnv/number-general).

Usage example:
```rust
use std::iter::FromIterator;
use afarray::Array;
use number_general::Number;

let a = [1, 2, 3];
let b = [5];

let product = &Array::from(&a[..]) * &Array::from(&b[..]);
assert_eq!(product, Array::from_iter(vec![5, 10, 15]));
assert_eq!(product.sum(), Number::from(30))
```

Note that you must have [ArrayFire](http://arrayfire.org) installed in order to build this crate. The installation
instructions are at [http://arrayfire.org/docs/installing.htm](http://arrayfire.org/docs/installing.htm).

In particular, you will need to make the ArrayFire package discoverable to pkg-config by adding
a pkg-config on your system's $PKG_CONFIG_PATH, like this `arrayfire.pc` config file:
```
prefix=/usr
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib64

Name: arrayfire
Description: the ArrayFire library
Version: 3.8
Libs: -L${libdir}
```

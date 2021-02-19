# Initial DAPHNE Prototype [WIP]

Initial prototype of the DAPHNE system.
Takes a plain text file written in *DaphneDSL* as input, parses, optimizes, JIT-compiles, and executes it.
Prints script outputs to `stdout`.

## 0. Requirements

Please ensure that your system meets the following requirements before trying to build the prototype.

**(*)**
You can view the version numbers as an orientation rather than a strict requirement.
Newer versions should work as well, older versions might work as well.

##### Operating system

| OS | distribution/version known to work (*) |
| --- | --- |
| GNU/Linux | Ubuntu 20.4.1 with kernel 5.8.0-43-generic |

##### Software

| tool/lib | version known to work (*) |
| ----------- | ----------- |
| clang | 10.0.0 |
| cmake | 3.16.3 |
| git | 2.25.1 |
| lld | 10.0.0 |
| ninja | 1.10.0 |
| pkg-config | 0.29.1 |
| uuid-dev |  |

##### Hardware

  - about 2.1 GB of free disk space (mostly due to MLIR/LLVM)

## 1. Obtain the source code

The prototype is based on MLIR, which is a part of the LLVM monorepo.
The LLVM monorepo is included in this repository as a submodule.
Thus, clone this repository as follows to also clone the submodule:

```bash
git clone --recursive https://gitlab.know-center.tugraz.at/daphne/prototype.git
```

Upstream changes to this repository might contain changes to the submodule (we might have upgraded to a newer version of MLIR/LLVM).
Thus, please pull as follows:

```bash
# in git >= 2.14
git pull --recurse-submodules

# in git < 2.14
git pull && git submodule update --init --recursive

# or use this little convenience script
./pull.sh
```

## 2. Build

Simply build the prototype using the build-script without any arguments:

```bash
./build.sh
```

When you do this the first time, or when there were updates to the LLVM submodule, this will also download and build the third-party material, which might increase the build time significantly.
Subsequent builds, e.g., when you changed something in this repository, will be much faster.

## 3. Run

Write a little DaphneDSL script or use `example.daphne`...

```
def main() {
    let x = 1;
    let y = 2;
    print(x + y);

    let m = rand(2, 3, 0, 1.0);
    print(m);
    print(m + m);
}
```

... and execute it as follows: `build/bin/daphnec example.daphne`.
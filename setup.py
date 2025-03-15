import platform
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

def get_cpu_flags():
    """Retrieve CPU flags to check for AVX support."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if line.startswith("flags"):
                    return line.split(":")[1].strip().split()
        except Exception as e:
            print(f"Error reading /proc/cpuinfo: {e}")
    else:
        print(f"Platform not supported: {platform.system()}")
    return []

def detect_avx_support():
    """Detect AVX support based on CPU flags and return the appropriate macro."""
    flags = get_cpu_flags()
    if "avx512f" in flags:
        return "__AVX512__"  # AVX512 support
    elif "avx2" in flags:
        return "__AVX256__"  # AVX2 support
    return None  # Scalar operations if no AVX support

# Determine the macro based on CPU capabilities
avx_macro = detect_avx_support()

# Set up compilation arguments
extra_compile_args = ["-fopenmp", "-march=native"]
if avx_macro:
    extra_compile_args.append(f"-D{avx_macro}")

setup(
    name="cpu_adam",
    version="0.0.1",
    author="JosephLiaw",
    description="CPU Adam optimizer with automatic AVX detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CppExtension(
            name="cpu_adam.cpu_adam_interface",
            sources=[
                "src/cpu_adam/cpu_adam_interface.cpp",
                "src/cpu_adam/cpu_adam_impl.cpp"
            ],
            include_dirs=["src/cpu_adam/includes"],
            extra_compile_args=extra_compile_args,
            extra_link_args=["-fopenmp"]
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"]
)
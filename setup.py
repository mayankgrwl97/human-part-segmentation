import setuptools

setuptools.setup(
    name="human-part-segmentation",
    version="1.0",
    author="Mayank Agarwal",
    author_email="mayankgrwl97@gmail.com",
    description="Human Part Segmentation",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['modules/src/*.h', 'modules/src/*.cpp',
                       'modules/src/*.cu', 'modules/src/utils/*.h',
                       'modules/src/utils/*.cuh']}
)

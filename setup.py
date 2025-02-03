import setuptools

scripts = [] #["./bin/prepare_maps"]

setuptools.setup(
    name="ksz4",
    version="0.0.1",
    author="Niall MacCrann",
    author_email="nm746@cam.ac.uk",
    description="",
    packages=["ksz4"],
    include_package_data=True,
    package_data={'ksz4': ['fg_term_defaults.yaml']},
    scripts=scripts
)

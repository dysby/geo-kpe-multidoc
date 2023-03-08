from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="geo_kpe_multidoc",
    version="0.1.0",
    description="Keyphrase Extraction with geospacial associations in multidocuments",
    author="dysby",
    author_email="dysby@example.com",
    scripts=["scripts/run.py"],
    # entry_points={
    #    'console_scripts': ['geo-kpe-multidoc-run=scripts.run:main']
    # }
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="natural-language-processing information-retrieval key-phrases key-phrase-extraction",
)

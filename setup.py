from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="lambre",
    version="2.0.0",
    description="a tool to measure the grammatical well-formedness of multilingual texts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adithya7/lambre",
    author="Adithya Pratapa, Antonios Anastasopoulos, Shruti Rijhwani, Aditi Chaudhary, David R. Mortensen, Graham Neubig and Yulia Tsvetkov",
    license="MIT",
    keywords="multilingual text-generation evaluation",
    packages=find_packages(),
    install_requires=["stanza==1.3.0", "scipy", "pyconll"],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["lambre=lambre.metric:main", "lambre-download=lambre.download:main"]},
)

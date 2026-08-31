import setuptools


with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="ct2foam",
	version="2.0.0",
	author="Heikki Kahila",
	author_email="heikki.kahila@gmail.com",
	description="A package converting Cantera thermophysical data into OpenFOAM dictionary format.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=["ct2foam", "ct2foam.tests"],
	include_package_data=True,
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: OS Independent",
	],
	python_requires=">=3.11",
	install_requires=[
		"cvxopt>=1.2.0",
		"numpy>=1.20",
		"scipy>=1.5.2",
		"matplotlib>=3.3",
	],
	entry_points={
		"console_scripts": [
			"ct2foam=ct2foam.ct2foam:main",
		],
	},
)

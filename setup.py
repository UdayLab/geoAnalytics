import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'geoanalytics',
    version = '2025.06.04.1',
    author = 'Rage Uday Kiran',
    author_email = 'uday.rage@gmail.com',
    description = 'This software is being developed at the University of Aizu, Aizu-Wakamatsu, Fukushima, Japan',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages=setuptools.find_packages(),
    url = 'https://github.com/udayRage/geoAnalytics',
    license='GPLv3',
    install_requires=[            # All necessary packages utilized by our PAMI software
        'psutil',
        'pandas',
        'matplotlib',
        'resource',
        'validators',
        'urllib3',
        'psycopg2-binary',
        'shapely',
        'tqdm',
        'mplcursors',
        'sklearn'
    ],
    classifiers = [
        'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Programming Language :: Python :: 3',
	'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires = '>=3.5',
)

from setuptools import setup, find_packages


setup(
    name='chatgpt-wrapper',
    version='0.0.1',
    description='A wrapper for chatgpt with logging, caching and more pythonic output parsing',
    long_description_content_type='text/markdown',
    url='https://github.com/Globe-Knowledge-Solutions/chatgpt',
    author='Ivan Yevenko',
    author_email='ivan@globe.engineer',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'openai',
        'diskcache',
    ]
)
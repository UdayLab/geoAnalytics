echo "Uninstall old geoAnalytics version"
pip uninstall -y geoanalytics
rm -rf dist/ geoanalytics.egg-info/ build/


echo "Running setup"
python3 setup.py sdist bdist_wheel

echo "Uploading to test repository"
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

echo "Wait for 5 minute to update the repository"
sleep 100

echo "installing geoAnalytics from the testPYPI"
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps geoanalytics

echo "Uploading geoAnalytics  to main PYPI repository"
python3 -m twine upload dist/*

echo "Deleting unnecessary files"
rm -rf dist/ geoanalytics.egg-info/ build/


echo "Completed."

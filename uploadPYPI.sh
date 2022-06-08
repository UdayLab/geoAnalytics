echo "Uninstall old geoAnalytics version"
pip uninstall -y geoAnalytics
rm -rf dist/ geoAnalytics.egg-info/ build/


echo "Running setup"
python3 setup.py sdist bdist_wheel

echo "Uploading to test repository"
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

echo "Wait for 5 minute to update the repository"
sleep 300

echo "installing geoAnalytics from the testPYPI"
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps geoAnalytics

echo "Uploading geoAnalytics  to main PYPI repository"
python3 -m twine upload dist/*

echo "Deleting unnecessary files"
rm -rf dist/ geoAnalytics.egg-info/ build/


echo "Completed."
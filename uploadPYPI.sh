echo "Uninstalling old geoAnalytics version"
pip uninstall -y geoanalytics
rm -rf dist/ geoanalytics.egg-info/ build/

echo "Installing build and twine if not already present"
python3 -m pip install --upgrade build twine

echo "Building source and wheel distributions"
python3 -m build

echo "Uploading to TestPyPI"
python3 -m twine upload --repository testpypi dist/*

echo "Waiting for TestPyPI to be ready"
RETRIES=5
for i in $(seq 1 $RETRIES); do
    echo "Attempt $i: Installing from TestPyPI..."
    python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps geoanalytics && break
    sleep 10
done

echo "Uploading to main PyPI"
python3 -m twine upload dist/*

echo "Cleaning up"
rm -rf dist/ geoanalytics.egg-info/ build/

echo "Completed."


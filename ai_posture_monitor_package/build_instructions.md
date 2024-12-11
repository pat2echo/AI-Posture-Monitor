### 1. **Package and Install Locally**
1. **Open the terminal in PyCharm** and navigate to the project root directory.
2. Run the following commands:
   - Build the package:
     ```bash
     python setup.py sdist
     ```
   - Install it locally for testing:
     ```bash
     pip install .
     ```

---

### 2. **Publish to PyPI (Optional)**
1. **Install the `twine` tool**:
   ```bash
   pip install twine
   ```
2. **Upload your package**:
   - First, create distribution files:
     ```bash
     python setup.py sdist bdist_wheel
     ```
   - Then, upload them to PyPI:
     ```bash
     twine upload dist/*
     ```

---

### 3. **Use the Package**
You can now import your package in Python:
```python
from ai_posture_monitor import PoseEstimation
```
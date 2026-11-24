### 📌 Description

Provide a clear, concise description of the pull request's purpose and the changes made.

### 📌 Type of change

Please delete options that are not relevant (and this line).

- Bug fix (non-breaking change which fixes an issue)
- New feature (non-breaking change which adds functionality)
- Breaking change (fix or feature that would cause existing functionality to not work as expected)

### 🚨 Motivation and Context

Why is this change required? Explain the problem it solves or the feature it adds.

### ✅ Checklist

See the [contributing guide](https://spherical-hci.readthedocs.io/en/latest/contributing/index.html) for details on each item.

- [ ] The change was discussed in an issue first, or it is a small bug fix or documentation correction.
- [ ] The pull request targets `develop` and does one thing.
- [ ] I added or updated tests in `tests/pipeline/` or `tests/database/`, and marked tests that reach a live service with `remote_data`.
- [ ] `pytest` (or `pixi run -e dev test`) passes.
- [ ] `ruff check src/ tests/` (or `pixi run -e dev lint`) reports no errors in the files I touched.
- [ ] If I changed `spherical.database`, `import spherical` still works without the `pipeline` extra.
- [ ] I updated the documentation where the change affects how the package is used.
- [ ] If I changed a reduction step, I describe below which sequence I reduced end to end and which products I checked.
- [ ] I ran the code myself and can explain every change, including any written with AI tools.

### 🔍 How to test/review

Describe steps for testing or reviewing your PR:

1. Step one
2. Step two
3. ...

### 📸 Screenshots (if applicable)

Add screenshots or GIFs that show your feature working or illustrate changes.

### 🔖 Related Issues/Tickets

List any related issues, tickets, or tasks here:
- Closes #<issue_number>
- Relates to #<issue_number>

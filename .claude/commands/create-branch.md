Create a new git branch following the project naming convention.

Format: `<type>/<developer>/<description>`

Arguments: $ARGUMENTS should be in format: `<type> <description>`
Example: `/create-branch feature add-caching` or `/create-branch fix login-error`

Steps:
1. Parse the type and description from arguments
2. Get the developer name from git config (user.name or user.email prefix)
3. Create branch: `git checkout -b <type>/<developer>/<description>`
4. Confirm the branch was created

Valid types: feature, fix, docs, refactor, test, chore

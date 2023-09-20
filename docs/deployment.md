# API Deployment Documentation

This documentation provides instructions on deploying the API to different environments, including non-production and production environments.

## Non-Production Environment Deployment

The non-production environment is automatically deployed upon merging changes into the `main` branch. This environment is typically used for development, testing, and staging purposes.

### Deployment Steps

To deploy the API to the non-production environment, follow these steps:

1. Make the necessary changes and enhancements to the codebase.
2. Create a feature branch for your changes and push your commits to that branch.
3. Open a pull request to merge your changes into the `main` branch.
4. Once the pull request is reviewed and approved, merge it into the `main` branch.
5. The deployment process will automatically trigger, pulling the latest changes from the `main` branch and deploying the API to the non-production environment.

## Production Environment Deployment

The production environment is the live environment where the API serves real users. To deploy to the production environment, specific steps need to be followed.

### Deployment Steps

To deploy the API to the production environment, follow these steps:

1. Ensure that all necessary changes and enhancements have been tested and verified in the non-production environment.
2. Bump the version number of the API following the format `v*`, where `*` represents the version number. For example, `v0.0.1`.
3. Create a new tag in the version control system, specifying the version number. For example, using Git: `git tag v0.0.1`.
4. Push the newly created tag to the remote repository: `git push origin v0.0.1`.
5. The deployment process will automatically trigger, pulling the tagged release from the repository and deploying the API to the production environment.

## Conclusion

By following the steps outlined in this documentation, you can easily deploy the API to both non-production and production environments. The non-production environment is automatically deployed upon merging changes to the `main` branch, while the production environment is deployed by creating version tags.
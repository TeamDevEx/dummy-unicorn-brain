# Contributing to Generative AI API

Thank you for your interest in supporting the Generative AI API! There is always lots to do as we continue to improve the capabilities of this API.

## Issues or feature requests

To submit an issue encountered while using the Generative AI API, please visit the Issues tab of the GitHub repository [here](https://github.com/telus/gen-ai-api/issues). Click the `New issue` button and you will be given the choice to either fill out an issue report, or provide feedback on features you would like to see implemented.

### Issue reports

To fill out an issue report, there are several pieces of information you will need to provide in order for the development team to be able to assist you in fixing the issue:

1. **Description**: Clear and concise description of the what the bug you are encountering is
2. **Steps to reproduce**: A list of actions that lead up to encountering the bug. If you are unable to reproduce the bug, make a note of that along with the steps actions that lead to the bug initially
3. **Expected behaviour**: Clear and concise description of what you were expecting to happen before encountering the bug
4. **Screenshots**: Any applicable screen shots that could be useful for triaging the issue
5. **Additional context**: Any additional information that could be useful in triaging this issue

### Feature requests

Filling out the feature request is similar to the issue reporting process. You will need the following information:

1. **Description of the problem**: Optional field; fill out any issues that are the motivation behind this feedback
2. **Desired outcome**: Clear and concise description of the feature you would like to see implemented
3. **Alternatives**: Any alternative solutions or features you have considered
4. **Additional context**: Any additional information that could be useful for developing this new feature

After filling out these forms, click 'Submit new issue' and a member from the development team will begin looking into the feedback

## Developing features

If you are interested in developing features and functionality for the Generative AI API, there is a full list of ongoing and planned tasks available in the project [epic](https://github.com/orgs/telus/projects/400/views/3). If you are interested in picking up any of these tasks and developing the feature, feel free to reach out to Julian Joseph or Liz Lozinsky before starting any work.

### Workflow

Members of the `gen-ai-squad` team on GitHub will have access to clone this repository instead of forking it as described below. If you are not a member of that team on GitHub, follow the steps described below.

1. **Set up local environment**: Fork this repository and make sure all of the dependencies in `requirements.txt` are installed. Copy the `.env.template` file and fill in the values that empty. To acquire an OpenAI key, reach out in the `#g-unicorn-ai` channel for assistance
2. **Run locally**: Follow the instructions in `README.md` for more detailed instructions on how to run the API locally
3. **Develop functionality**: Implement the functionality according to the requirements of the accompanying task; fully test implementation and ensure code adheres to standards
4. **Open pull request**: Push code to new branch and open a pull request in the forked repository. Fill out the information in the template. Assign at least one reviewer from the `gen-ai-squad` on GitHub to review the changes that were made. Complete any changes that are requested from the reviewers

Once the pull request has been merged, the team will handle the non-prod and production environments. Congratulations on deploying a feature to the Generative AI API! :tada:

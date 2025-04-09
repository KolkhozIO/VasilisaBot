# Contributing to VasilisaBot

Thank you for your interest in contributing to VasilisaBot! This document provides guidelines and instructions for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/vasilisabot.git`
3. Create a new branch for your feature or bugfix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-bugfix-name`
4. Install dependencies: `pip install -r requirements.txt`
5. Make your changes
6. Test your changes thoroughly
7. Commit your changes: `git commit -m "Add your meaningful commit message"`
8. Push to your fork: `git push origin feature/your-feature-name`
9. Create a Pull Request from your fork to the main repository

## Development Environment

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and configure your environment variables

## Code Style

Please follow these coding standards:

- Use PEP 8 style guide for Python code
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write clear, descriptive commit messages

## Testing

Before submitting a pull request, please test your changes thoroughly:

1. Test with different Ollama models
2. Test in both private and group chats
3. Test all affected functionality

## Pull Request Process

1. Update the README.md and/or documentation with details of changes if appropriate
2. The PR should work for both Python 3.8+ environments
3. The PR will be merged once it receives approval from maintainers

## Feature Requests and Bug Reports

Please use the GitHub Issues section to submit feature requests and bug reports. Include as much detail as possible:

- For bugs: steps to reproduce, expected behavior, actual behavior, and environment details
- For features: clear description of the feature and its benefits

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
# Project guidelines

- Follow standard coding best practices and the Ponytail skill’s guidance when available. Prefer the simplest correct solution and reuse existing code where practical.
- Check the relevant code and documentation first. Ask the user for more information when uncertainty remains and the answer could materially affect the work.
- Keep changes focused on the request. Avoid unrelated refactors, formatting changes, and new dependencies unless they are needed for the task.
- Verify changes with the smallest relevant check. Report any verification you could not complete.
- Make the code and comments look clean and final. Never write instructions to the user directly in code comments, only through the chat.
- Let the user do the testing and git commits.

## Cafe Cup Picking

- Use `README.txt` and `DATASET_GUIDE.txt` for project setup, data conventions, and submission requirements.
- Keep `baseline.ipynb` and `weights/baseline.pth` intact for comparison. Develop model changes in `improved.ipynb` using all four views.
- Keep the `cafe.inference` cells self-contained and runnable without downloads, login, or personal Drive access.

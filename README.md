# Meal Planner Lite


1. It should be runnable from the command line: python app.py --prompt "example user request :white_check_mark: 
2. runnable from CLI --temperature, --top_p, --max_tokens parametreleri CLI  :white_check_mark: 

<img width="1191" height="1052" alt="image" src="https://github.com/user-attachments/assets/1ad0e71b-6364-4c3b-b79d-b902a05fa958" />


3. If the JSON output is corrupted, a one-time repair attempt should be made. :white_check_mark: 
```
 def _repair_json(self, text: str) -> Dict[str, Any]:
    """
    Attempt to repair malformed JSON output
    """
    # step 1: clean Markdown code blocks
    text = text.replace("```json", "").replace("```", "").strip()
    
    # step 2: try again parsing 
    try:
        plan = json.loads(text)
        self._validate_plan(plan)
        print("[INFO] JSON repaired successfully")
        return plan
    except Exception as e:
        # step 3: give error if fail to repair
        print(f"[ERROR] Could not repair JSON: {e}", file=sys.stderr)
        raise ValueError("Failed to generate valid plan JSON")

```

4. Prepare at least 3 different input examples and put them in the examples/inputs.txt file. :white_check_mark:

5. Develop the application using a chain structure with LangChain. :white_check_mark:

<img width="1191" height="1052" alt="image" src="https://github.com/user-attachments/assets/67055e70-6745-45ca-a98b-2165fc40f0e7" />

<img width="1191" height="1052" alt="image" src="https://github.com/user-attachments/assets/e32b4b03-71ad-46aa-a43f-43b16e86c72f" />

6. Flask-ui (not done fully)
   <img width="1351" height="803" alt="image" src="https://github.com/user-attachments/assets/e6acccce-2428-4b27-9b3d-55260b392674" />


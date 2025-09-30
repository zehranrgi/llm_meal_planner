#!/usr/bin/env python3
"""
Recipe Optimizer - Web Interface
Beautiful web UI with Flask
Usage: python app_web.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import json
import os
import argparse
import time
from openai import OpenAI

app = Flask(__name__)

# Disable cache in debug mode
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recipe Optimizer - Prompt Chaining</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-section label {
            display: block;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            font-size: 1.1em;
        }
        
        .input-section textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s;
        }
        
        .input-section textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .settings {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .setting {
            display: flex;
            flex-direction: column;
        }
        
        .setting label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #555;
        }
        
        .setting select, .setting input {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        .setting select:focus, .setting input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .submit-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        .submit-btn:active {
            transform: translateY(0);
        }
        
        .submit-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            display: none;
            margin-top: 40px;
        }
        
        .result-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
        }
        
        .result-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .plan-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        
        .plan-item h3 {
            color: #764ba2;
            margin-bottom: 10px;
        }
        
        .plan-item ul {
            list-style: none;
            padding-left: 0;
        }
        
        .plan-item li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }
        
        .plan-item li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
        }
        
        .markdown-content {
            background: white;
            padding: 25px;
            border-radius: 10px;
            line-height: 1.8;
        }
        
        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
            color: #333;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        
        .markdown-content ul, .markdown-content ol {
            margin-left: 20px;
            margin-bottom: 15px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .example-prompts {
            background: #f0f4ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .example-prompts h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .example-btn {
            display: inline-block;
            padding: 8px 15px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 20px;
            margin: 5px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
        }
        
        .example-btn:hover {
            background: #667eea;
            color: white;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .settings {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🍳 Recipe Optimizer</h1>
            <p>AI-Powered Meal Planning with Prompt Chaining</p>
        </div>
        
        <div class="main-content">
            <div class="example-prompts">
                <h3>Try These Examples:</h3>
                <span class="example-btn" data-prompt="5 vegetarian dinners under 30 minutes">Quick Vegetarian</span>
                <span class="example-btn" data-prompt="7-day meal plan for muscle building, 2500 cal/day">Muscle Building</span>
                <span class="example-btn" data-prompt="10 family dinners for 4 people, 100 dollar budget">Family Budget</span>
            </div>
            
            <div class="input-section">
                <label for="prompt">Your Recipe Request:</label>
                <textarea id="prompt" placeholder="Example: 5 vegetarian dinners under 30 minutes, budget-friendly"></textarea>
            </div>
            
            <div class="settings">
                <div class="setting">
                    <label for="provider">Provider:</label>
                    <select id="provider">
                        <option value="openai">OpenAI</option>
                        <option value="openrouter">OpenRouter</option>
                    </select>
                </div>
                
                <div class="setting">
                    <label for="model">Model:</label>
                    <select id="model">
                        <option value="gpt-4-turbo-preview">GPT-4 Turbo</option>
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    </select>
                </div>
                
                <div class="setting">
                    <label for="temperature">Temperature:</label>
                    <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1">
                </div>
                
                <div class="setting">
                    <label for="max_tokens">Max Tokens:</label>
                    <input type="number" id="max_tokens" value="1500" min="100" max="4000" step="100">
                </div>
            </div>
            
            <button class="submit-btn" id="generateBtn">Generate Recipe Plan</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Generating your personalized plan...</p>
            </div>
            
            <div class="results" id="results">
                <div class="result-section">
                    <h2>📋 Structured Plan</h2>
                    <div id="plan-output"></div>
                </div>
                
                <div class="result-section">
                    <h2>📝 Final Response</h2>
                    <div class="markdown-content" id="response-output"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script type="text/javascript">
        console.log("JavaScript loading...");
        
        // Wait for DOM to be ready
        document.addEventListener('DOMContentLoaded', function() {
            console.log("DOM loaded, initializing...");
            
            // Example prompt buttons
            const exampleButtons = document.querySelectorAll('.example-btn');
            exampleButtons.forEach(function(btn) {
                btn.addEventListener('click', function() {
                    const prompt = this.getAttribute('data-prompt');
                    console.log("Example clicked:", prompt);
                    document.getElementById('prompt').value = prompt;
                });
            });
            
            // Provider change handler
            const providerSelect = document.getElementById('provider');
            providerSelect.addEventListener('change', function() {
                console.log("Provider changed to:", this.value);
                updateModelOptions();
            });
            
            // Generate button handler
            const generateBtn = document.getElementById('generateBtn');
            generateBtn.addEventListener('click', function() {
                console.log("Generate button clicked!");
                generatePlan();
            });
            
            console.log("All event listeners attached!");
        });
        
        function updateModelOptions() {
            console.log("updateModelOptions called");
            const provider = document.getElementById('provider').value;
            const modelSelect = document.getElementById('model');
            
            if (provider === 'openrouter') {
                modelSelect.innerHTML = '<option value="openai/gpt-4-turbo">OpenAI GPT-4 Turbo</option><option value="openai/gpt-3.5-turbo">OpenAI GPT-3.5 Turbo</option><option value="anthropic/claude-3-sonnet">Claude 3 Sonnet</option><option value="meta-llama/llama-3-70b-instruct">Llama 3 70B</option>';
            } else {
                modelSelect.innerHTML = '<option value="gpt-4-turbo-preview">GPT-4 Turbo</option><option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>';
            }
        }
        
        async function generatePlan() {
            console.log("generatePlan called!");
            
            const prompt = document.getElementById('prompt').value;
            const provider = document.getElementById('provider').value;
            const model = document.getElementById('model').value;
            const temperature = parseFloat(document.getElementById('temperature').value);
            const max_tokens = parseInt(document.getElementById('max_tokens').value);
            
            console.log("Parameters:", {prompt, provider, model, temperature, max_tokens});
            
            if (!prompt.trim()) {
                alert('Please enter a recipe request');
                return;
            }
            
            const submitBtn = document.querySelector('.submit-btn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            submitBtn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';
            
            try {
                console.log("Sending request to /generate...");
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        provider: provider,
                        model: model,
                        temperature: temperature,
                        max_tokens: max_tokens
                    })
                });
                
                console.log("Response status:", response.status);
                const data = await response.json();
                console.log("Response data received");
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    console.error("Server error:", data.error);
                    return;
                }
                
                displayResults(data);
            } catch (error) {
                console.error("Error occurred:", error);
                alert('Error: ' + error.message);
            } finally {
                submitBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        function displayResults(data) {
            console.log("displayResults called");
            const planOutput = document.getElementById('plan-output');
            const responseOutput = document.getElementById('response-output');
            
            let planHTML = '';
            
            planHTML += '<div class="plan-item"><h3>Steps</h3><ul>';
            data.plan.steps.forEach(function(step) {
                planHTML += '<li>' + step + '</li>';
            });
            planHTML += '</ul></div>';
            
            planHTML += '<div class="plan-item"><h3>Assumptions</h3><ul>';
            data.plan.assumptions.forEach(function(assumption) {
                planHTML += '<li>' + assumption + '</li>';
            });
            planHTML += '</ul></div>';
            
            planHTML += '<div class="plan-item"><h3>Success Criteria</h3><ul>';
            data.plan.success_criteria.forEach(function(criterion) {
                planHTML += '<li>' + criterion + '</li>';
            });
            planHTML += '</ul></div>';
            
            planOutput.innerHTML = planHTML;
            
            const markdownHTML = data.final_response
                .replace(/### (.*)/g, '<h3>$1</h3>')
                .replace(/## (.*)/g, '<h2>$1</h2>')
                .replace(/# (.*)/g, '<h1>$1</h1>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n\n/g, '</p><p>')
                .replace(/\n/g, '<br>');
            
            responseOutput.innerHTML = '<p>' + markdownHTML + '</p>';
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""

class PromptChainApp:
    def __init__(self, api_key, base_url=None, model="gpt-4-turbo-preview", 
                 temperature=0.7, max_tokens=2000):
        self.api_key = api_key
        client_params = {"api_key": self.api_key}
        if base_url:
            client_params["base_url"] = base_url
        self.client = OpenAI(**client_params)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def call_llm(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
    
    def generate(self, user_input):
        # Stage 1: Planning
        planning_prompt = """You are a planning assistant. Create a JSON plan with:
- "steps": array of 3-6 actionable steps
- "assumptions": array of 2-4 assumptions
- "success_criteria": array of 2-3 success criteria
Output ONLY valid JSON, no markdown formatting."""
        
        plan_text = self.call_llm(planning_prompt, f"User Request: {user_input}\n\nCreate plan:")
        plan_text = plan_text.replace("```json", "").replace("```", "").strip()
        
        # Extract JSON
        start = plan_text.find("{")
        end = plan_text.rfind("}") + 1
        if start != -1 and end > start:
            plan_text = plan_text[start:end]
        
        plan = json.loads(plan_text)
        
        # Stage 2: Execution
        execution_prompt = """Create a comprehensive Markdown response with headings, 
bullet points, and a "Next Steps" section."""
        
        final_response = self.call_llm(
            execution_prompt,
            f"Original: {user_input}\n\nPlan: {json.dumps(plan)}\n\nGenerate response:"
        )
        
        return {"plan": plan, "final_response": final_response}

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Debug: Log the request
        print("[DEBUG] Received generate request")
        
        data = request.json
        if not data:
            print("[ERROR] No JSON data received")
            return jsonify({"error": "No data received"}), 400
        
        print(f"[DEBUG] Request data: {data}")
        
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4-turbo-preview')
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 1500))
        prompt = data.get('prompt', '').strip()
        
        print(f"[DEBUG] Provider: {provider}, Model: {model}")
        print(f"[DEBUG] Prompt: {prompt[:50]}...")
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Get API key based on provider
        if provider == 'openrouter':
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            print(f"[DEBUG] Using OpenRouter")
        else:
            base_url = None
            api_key = os.getenv("OPENAI_API_KEY")
            print(f"[DEBUG] Using OpenAI")
        
        if not api_key:
            print("[ERROR] API key not found")
            return jsonify({"error": "API key not configured. Please set OPENAI_API_KEY or OPENROUTER_API_KEY in .env file"}), 500
        
        print(f"[DEBUG] API key found: {api_key[:10]}...")
        
        # Generate plan
        print("[DEBUG] Creating PromptChainApp instance")
        app_instance = PromptChainApp(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        print("[DEBUG] Generating response")
        result = app_instance.generate(prompt)
        
        print("[DEBUG] Success! Returning result")
        return jsonify(result)
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode error: {e}")
        return jsonify({"error": f"Invalid JSON in response: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

def main():
    parser = argparse.ArgumentParser(description="Recipe Optimizer Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Recipe Optimizer - Web Interface")
    print("="*60)
    print(f"Server starting on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
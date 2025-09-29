#!/usr/bin/env python3
"""
Prompt Chaining Application - Recipe Optimizer
Supports both OpenAI and OpenRouter APIs
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional
from openai import OpenAI


class PromptChainApp:
    """Two-stage prompt chaining application for recipe optimization"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.4,
        top_p: float = 1.0,
        max_tokens: int = 1200
    ):
        """
        Initialize the app with API credentials and parameters
        
        Args:
            api_key: OpenAI or OpenRouter API key
            base_url: Base URL (for OpenRouter use: https://openrouter.ai/api/v1)
            model: Model name
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        # Initialize OpenAI client (works with OpenRouter too)
        client_params = {"api_key": self.api_key}
        if self.base_url:
            client_params["base_url"] = self.base_url
        
        self.client = OpenAI(**client_params)
        
        print(f"[INFO] Initialized with model: {self.model}")
        if self.base_url:
            print(f"[INFO] Using custom endpoint: {self.base_url}")
    
    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM API with given prompts
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User input prompt
            
        Returns:
            Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"[ERROR] API Error: {e}", file=sys.stderr)
            raise
    
    def stage_1_planning(self, user_input: str) -> Dict[str, Any]:
        """
        Stage 1: Create a structured plan from user input
        
        Args:
            user_input: Raw user request
            
        Returns:
            Dictionary with steps, assumptions, and success_criteria
        """
        system_prompt = """You are a planning assistant. Your job is to break down user requests into actionable steps.

Given a user's request, create a JSON plan with exactly these fields:
- "steps": array of 3-6 clear, actionable steps
- "assumptions": array of 2-4 assumptions you're making
- "success_criteria": array of 2-3 measurable success criteria

Keep steps concise but specific. Focus on practical actions.
Output ONLY valid JSON, no markdown formatting or explanation."""

        user_prompt = f"""User Request: {user_input}

Create a structured plan in JSON format."""

        print("\n[STAGE 1] Generating Plan...")
        
        response = self.call_llm(system_prompt, user_prompt)
        
        # Try to parse JSON
        try:
            plan = json.loads(response)
            self._validate_plan(plan)
            print("[STAGE 1] Plan generated successfully")
            return plan
        
        except json.JSONDecodeError:
            print("[WARNING] JSON parse error, attempting repair...")
            return self._repair_json(response)
    
    def _validate_plan(self, plan: Dict[str, Any]) -> None:
        """Validate plan structure"""
        required_fields = ["steps", "assumptions", "success_criteria"]
        for field in required_fields:
            if field not in plan:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(plan[field], list):
                raise ValueError(f"Field {field} must be a list")
    
    def _repair_json(self, text: str) -> Dict[str, Any]:
        """
        Attempt to repair malformed JSON output
        
        Args:
            text: Potentially malformed JSON text
            
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Try parsing again
        try:
            plan = json.loads(text)
            self._validate_plan(plan)
            print("[INFO] JSON repaired successfully")
            return plan
        except Exception as e:
            print(f"[ERROR] Could not repair JSON: {e}", file=sys.stderr)
            raise ValueError("Failed to generate valid plan JSON")
    
    def stage_2_execution(self, user_input: str, plan: Dict[str, Any]) -> str:
        """
        Stage 2: Generate final response using the plan
        
        Args:
            user_input: Original user request
            plan: Structured plan from stage 1
            
        Returns:
            Markdown-formatted final response
        """
        system_prompt = """You are a helpful assistant that creates detailed, user-friendly responses.

You will receive:
1. The original user request
2. A structured plan with steps, assumptions, and success criteria

Your task:
- Create a comprehensive response in Markdown format
- Use clear headings and bullet points
- Include a "Next Steps" section at the end
- Be practical, friendly, and actionable
- Reference the plan steps naturally in your response"""

        user_prompt = f"""Original Request:
{user_input}

Plan:
{json.dumps(plan, indent=2)}

Generate a complete, user-friendly response in Markdown format."""

        print("\n[STAGE 2] Generating Final Response...")
        
        response = self.call_llm(system_prompt, user_prompt)
        
        print("[STAGE 2] Final response generated")
        return response
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Run the complete two-stage chain
        
        Args:
            user_input: User's initial request
            
        Returns:
            Dictionary with plan and final_response
        """
        print(f"\n{'='*60}")
        print(f"USER INPUT: {user_input}")
        print(f"{'='*60}")
        
        # Stage 1: Planning
        plan = self.stage_1_planning(user_input)
        
        # Stage 2: Execution
        final_response = self.stage_2_execution(user_input, plan)
        
        return {
            "user_input": user_input,
            "plan": plan,
            "final_response": final_response
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Prompt Chaining App - Recipe Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --prompt "5 vegetarian dinners under 30 minutes"
  python app.py --prompt "Meal plan for weight loss, 1500 cal/day" --temperature 0.5
  python app.py --provider openrouter --model "anthropic/claude-3-sonnet"
        """
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User input/request"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "openrouter"],
        default="openai",
        help="API provider (default: openai)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: gpt-4-turbo-preview for OpenAI, openai/gpt-4-turbo for OpenRouter)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (default: 1.0)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="Maximum tokens to generate (default: 2000)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save output to JSON file"
    )
    
    args = parser.parse_args()
    
    # Set defaults based on provider
    if args.provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        model = args.model or "openai/gpt-4-turbo"
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        base_url = None
        model = args.model or "gpt-4-turbo-preview"
        api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        # Initialize app
        app = PromptChainApp(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        # Run the chain
        result = app.run(args.prompt)
        
        # Display results
        print(f"\n{'='*60}")
        print("PLAN (JSON)")
        print(f"{'='*60}")
        print(json.dumps(result["plan"], indent=2, ensure_ascii=False))
        
        print(f"\n{'='*60}")
        print("FINAL RESPONSE (Markdown)")
        print(f"{'='*60}")
        print(result["final_response"])
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n[INFO] Results saved to {args.output}")
        
        print(f"\n{'='*60}")
        print("COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
    
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
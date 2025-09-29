#!/usr/bin/env python3
"""
Prompt Chaining with LangChain - Recipe Optimizer
Bonus Implementation
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Pydantic Models for Structured Output
class Plan(BaseModel):
    """Structured plan model"""
    steps: list[str] = Field(description="3-6 actionable steps")
    assumptions: list[str] = Field(description="2-4 assumptions")
    success_criteria: list[str] = Field(description="2-3 success criteria")


class PromptChainLangChain:
    """LangChain-based implementation of prompt chaining"""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.4,
        max_tokens: int = 1000
    ):
        """Initialize LangChain components"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key required")
        
        # Configure LLM
        llm_params = {
            "api_key": self.api_key,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if base_url:
            llm_params["base_url"] = base_url
        
        self.llm = ChatOpenAI(**llm_params)
        
        # Setup parsers
        self.plan_parser = PydanticOutputParser(pydantic_object=Plan)
        
        print(f"[INFO] LangChain initialized with model: {model}")
    
    def create_planning_chain(self) -> LLMChain:
        """Create Stage 1: Planning chain"""
        
        planning_template = """You are a planning assistant. Break down the user's request into a structured plan.

{format_instructions}

User Request: {user_input}

Generate a JSON plan with steps, assumptions, and success_criteria."""
        
        planning_prompt = ChatPromptTemplate.from_template(
            template=planning_template,
            partial_variables={
                "format_instructions": self.plan_parser.get_format_instructions()
            }
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=planning_prompt,
            output_key="plan_text"
        )
    
    def create_execution_chain(self) -> LLMChain:
        """Create Stage 2: Execution chain"""
        
        execution_template = """You are a helpful assistant creating detailed responses.

Original Request: {user_input}

Structured Plan:
{plan}

Create a comprehensive Markdown response that:
- Uses clear headings and bullet points
- References the plan steps naturally
- Includes a "Next Steps" section at the end
- Is practical and user-friendly

Generate the final response:"""
        
        execution_prompt = ChatPromptTemplate.from_template(execution_template)
        
        return LLMChain(
            llm=self.llm,
            prompt=execution_prompt,
            output_key="final_response"
        )
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """Execute the complete chain"""
        
        print(f"\n{'='*60}")
        print(f"USER INPUT: {user_input}")
        print(f"{'='*60}")
        
        # Stage 1: Planning
        print("\n[STAGE 1] Planning with LangChain...")
        planning_chain = self.create_planning_chain()
        
        try:
            plan_result = planning_chain.invoke({"user_input": user_input})
            plan_text = plan_result["plan_text"]
            
            # Parse the plan
            plan_dict = self.plan_parser.parse(plan_text)
            plan = plan_dict.dict()
            
            print("[STAGE 1] Plan generated successfully")
            
        except Exception as e:
            print(f"[WARNING] Pydantic parsing failed, falling back to JSON: {e}")
            # Fallback to simple JSON parsing
            try:
                plan_text = plan_text.replace("```json", "").replace("```", "").strip()
                plan = json.loads(plan_text)
            except:
                raise ValueError("Failed to parse plan from LLM output")
        
        # Stage 2: Execution
        print("\n[STAGE 2] Generating final response...")
        execution_chain = self.create_execution_chain()
        
        execution_result = execution_chain.invoke({
            "user_input": user_input,
            "plan": json.dumps(plan, indent=2)
        })
        
        final_response = execution_result["final_response"]
        print("[STAGE 2] Final response generated")
        
        return {
            "user_input": user_input,
            "plan": plan,
            "final_response": final_response
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Prompt Chaining with LangChain - Recipe Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help="API provider"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="Maximum tokens"
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
        app = PromptChainLangChain(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=args.temperature,
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
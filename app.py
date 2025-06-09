import streamlit as st
import openai
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import PyPDF2
import io
import json
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Report Validation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MetaPromptingAgent:
    """Base class for AI agents with meta prompting capabilities"""
    
    def __init__(self, agent_name: str, openai_api_key: str):
        self.agent_name = agent_name
        self.llm = OpenAI(temperature=0.2, openai_api_key=openai_api_key)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
    def meta_prompt_generator(self, task_description: str, validation_criteria: List[str]) -> str:
        """Generate optimized prompts using meta prompting techniques"""
        
        meta_prompt = f"""
        You are an expert prompt engineer and {self.agent_name} validation specialist. 
        Your task is to create the most effective prompt for validating financial statements.
        
        TASK: {task_description}
        
        VALIDATION CRITERIA:
        {chr(10).join(f"- {criteria}" for criteria in validation_criteria)}
        
        Using meta prompting principles, create a systematic validation framework that:
        1. Breaks down complex validation into manageable subtasks
        2. Uses structured thinking and self-reflection
        3. Implements recursive checking for accuracy
        4. Provides clear, actionable feedback
        5. Follows regulatory compliance requirements
        
        Generate an optimized prompt that will ensure thorough and accurate validation.
        """
        
        return meta_prompt
    
    def self_reflection_prompt(self, validation_result: str) -> str:
        """Generate self-reflection prompt to improve validation quality"""
        
        reflection_prompt = f"""
        Review your validation analysis below and perform self-reflection:
        
        VALIDATION RESULT:
        {validation_result}
        
        Self-reflection questions:
        1. Did I check all required disclosure items?
        2. Are my findings supported by specific evidence from the document?
        3. Have I considered materiality thresholds?
        4. Are there any inconsistencies I missed?
        5. Do my recommendations align with regulatory requirements?
        
        Provide an improved validation analysis addressing any gaps identified.
        """
        
        return reflection_prompt

class BalanceSheetAgent(MetaPromptingAgent):
    """AI Agent for Balance Sheet validation"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("Balance Sheet Validation", openai_api_key)
        self.validation_criteria = [
            "Current vs Non-current asset/liability classification",
            "Property, Plant and Equipment disclosures",
            "Investment classifications and fair value disclosures",
            "Trade receivables aging and bad debt provisions",
            "Share capital and reserves composition",
            "Borrowings classification and security details",
            "Related party transactions disclosure",
            "Contingent liabilities and commitments"
        ]
    
    def validate_balance_sheet(self, financial_text: str) -> Dict[str, Any]:
        """Validate balance sheet using meta prompting"""
        
        # Step 1: Generate optimized validation prompt
        meta_prompt = self.meta_prompt_generator(
            "Validate Balance Sheet compliance with Schedule III Division II requirements",
            self.validation_criteria
        )
        
        # Step 2: Initial validation
        validation_prompt = PromptTemplate(
            input_variables=["financial_text", "criteria"],
            template="""
            You are an expert financial analyst specializing in Balance Sheet validation.
            
            FINANCIAL STATEMENT TEXT:
            {financial_text}
            
            VALIDATION CRITERIA:
            {criteria}
            
            Perform systematic validation using this framework:
            
            1. STRUCTURAL ANALYSIS:
               - Verify current/non-current classification
               - Check line item presentations
               - Validate mathematical accuracy
            
            2. DISCLOSURE COMPLIANCE:
               - Assess mandatory disclosures per Schedule III
               - Check notes cross-references
               - Verify comparative figures
            
            3. REGULATORY REQUIREMENTS:
               - Materiality considerations
               - Rounding consistency
               - Related party disclosures
            
            4. QUALITY ASSESSMENT:
               - Identify missing information
               - Flag potential red flags
               - Provide improvement recommendations
            
            Provide detailed findings with specific line references.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=validation_prompt)
        initial_result = chain.run(
            financial_text=financial_text,
            criteria="\n".join(f"- {c}" for c in self.validation_criteria)
        )
        
        # Step 3: Self-reflection and improvement
        reflection_prompt_text = self.self_reflection_prompt(initial_result)
        reflection_chain = LLMChain(llm=self.llm, prompt=PromptTemplate(
            input_variables=["reflection"],
            template="{reflection}"
        ))
        
        final_result = reflection_chain.run(reflection=reflection_prompt_text)
        
        return {
            "agent": "Balance Sheet",
            "initial_analysis": initial_result,
            "refined_analysis": final_result,
            "compliance_score": self._calculate_compliance_score(final_result),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_compliance_score(self, analysis_text: str) -> int:
        """Calculate compliance score based on analysis"""
        # Simple scoring logic - can be enhanced
        positive_indicators = ["compliant", "adequate", "proper", "correct", "complete"]
        negative_indicators = ["missing", "incomplete", "non-compliant", "inadequate", "error"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_text.lower())
        negative_count = sum(1 for indicator in negative_indicators if indicator in analysis_text.lower())
        
        base_score = 50
        score = base_score + (positive_count * 10) - (negative_count * 15)
        return max(0, min(100, score))

class ProfitLossAgent(MetaPromptingAgent):
    """AI Agent for Profit & Loss validation"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("Profit & Loss Validation", openai_api_key)
        self.validation_criteria = [
            "Revenue recognition and classification",
            "Operating vs non-operating income segregation",
            "Expense categorization and disclosure",
            "Exceptional and extraordinary items",
            "Tax expense calculation and deferred tax",
            "Earnings per share computation",
            "Other comprehensive income items",
            "Related party transaction disclosures"
        ]
    
    def validate_profit_loss(self, financial_text: str) -> Dict[str, Any]:
        """Validate P&L using meta prompting"""
        
        validation_prompt = PromptTemplate(
            input_variables=["financial_text", "criteria"],
            template="""
            You are an expert financial analyst specializing in Statement of Profit and Loss validation.
            
            FINANCIAL STATEMENT TEXT:
            {financial_text}
            
            VALIDATION FRAMEWORK:
            
            1. REVENUE ANALYSIS:
               - Verify revenue from operations classification
               - Check other income segregation
               - Assess revenue recognition policies
            
            2. EXPENSE VALIDATION:
               - Employee benefits expense breakdown
               - Finance costs classification
               - Depreciation and amortization
               - Other expenses categorization
            
            3. PROFIT COMPUTATION:
               - Exceptional items treatment
               - Tax expense validation
               - Discontinued operations
               - Other comprehensive income
            
            4. DISCLOSURE REQUIREMENTS:
               - Earnings per share calculation
               - Additional information notes
               - Prior period comparatives
            
            VALIDATION CRITERIA:
            {criteria}
            
            Provide systematic analysis with compliance assessment.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=validation_prompt)
        result = chain.run(
            financial_text=financial_text,
            criteria="\n".join(f"- {c}" for c in self.validation_criteria)
        )
        
        return {
            "agent": "Profit & Loss",
            "analysis": result,
            "compliance_score": self._calculate_compliance_score(result),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_compliance_score(self, analysis_text: str) -> int:
        """Calculate compliance score"""
        # Implementation similar to BalanceSheetAgent
        positive_indicators = ["compliant", "adequate", "proper", "correct", "complete"]
        negative_indicators = ["missing", "incomplete", "non-compliant", "inadequate", "error"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_text.lower())
        negative_count = sum(1 for indicator in negative_indicators if indicator in analysis_text.lower())
        
        base_score = 50
        score = base_score + (positive_count * 10) - (negative_count * 15)
        return max(0, min(100, score))

class CashFlowAgent(MetaPromptingAgent):
    """AI Agent for Cash Flow Statement validation"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("Cash Flow Validation", openai_api_key)
        self.validation_criteria = [
            "Operating activities cash flow presentation",
            "Investing activities classification",
            "Financing activities segregation",
            "Reconciliation with net income",
            "Non-cash transactions disclosure",
            "Cash and cash equivalents definition",
            "Foreign exchange impact"
        ]
    
    def validate_cash_flow(self, financial_text: str) -> Dict[str, Any]:
        """Validate cash flow statement using meta prompting"""
        
        validation_prompt = PromptTemplate(
            input_variables=["financial_text", "criteria"],
            template="""
            You are an expert financial analyst specializing in Cash Flow Statement validation.
            
            FINANCIAL STATEMENT TEXT:
            {financial_text}
            
            VALIDATION FRAMEWORK:
            
            1. OPERATING ACTIVITIES:
               - Direct vs indirect method
               - Working capital changes
               - Non-cash adjustments
            
            2. INVESTING ACTIVITIES:
               - Capital expenditure
               - Investment transactions
               - Asset disposals
            
            3. FINANCING ACTIVITIES:
               - Borrowing activities
               - Equity transactions
               - Dividend payments
            
            4. RECONCILIATION & DISCLOSURE:
               - Opening/closing cash reconciliation
               - Non-cash transactions
               - Restricted cash disclosure
            
            CRITERIA: {criteria}
            
            Provide comprehensive validation analysis.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=validation_prompt)
        result = chain.run(
            financial_text=financial_text,
            criteria="\n".join(f"- {c}" for c in self.validation_criteria)
        )
        
        return {
            "agent": "Cash Flow",
            "analysis": result,
            "compliance_score": self._calculate_compliance_score(result),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_compliance_score(self, analysis_text: str) -> int:
        """Calculate compliance score"""
        positive_indicators = ["compliant", "adequate", "proper", "correct", "complete"]
        negative_indicators = ["missing", "incomplete", "non-compliant", "inadequate", "error"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_text.lower())
        negative_count = sum(1 for indicator in negative_indicators if indicator in analysis_text.lower())
        
        base_score = 50
        score = base_score + (positive_count * 10) - (negative_count * 15)
        return max(0, min(100, score))

class NotesAgent(MetaPromptingAgent):
    """AI Agent for Notes to Financial Statements validation"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("Notes Validation", openai_api_key)
        self.validation_criteria = [
            "Accounting policies disclosure",
            "Significant estimates and judgments",
            "Property, plant and equipment details",
            "Investment classification and valuation",
            "Borrowings terms and conditions",
            "Related party relationships and transactions",
            "Contingent liabilities and commitments",
            "Subsequent events disclosure"
        ]
    
    def validate_notes(self, financial_text: str) -> Dict[str, Any]:
        """Validate notes using meta prompting"""
        
        validation_prompt = PromptTemplate(
            input_variables=["financial_text", "criteria"],
            template="""
            You are an expert financial analyst specializing in Notes to Financial Statements validation.
            
            FINANCIAL STATEMENT TEXT:
            {financial_text}
            
            VALIDATION FRAMEWORK:
            
            1. ACCOUNTING POLICIES:
               - Revenue recognition policy
               - Depreciation methods
               - Inventory valuation
               - Investment classification
            
            2. DETAILED DISCLOSURES:
               - Asset breakdowns and reconciliations
               - Liability terms and conditions
               - Equity movements
               - Income and expense analysis
            
            3. REGULATORY COMPLIANCE:
               - Related party disclosures
               - Contingent liabilities
               - Commitments
               - Subsequent events
            
            4. ADEQUACY ASSESSMENT:
               - Information completeness
               - Clarity and understandability
               - Cross-reference accuracy
            
            CRITERIA: {criteria}
            
            Provide detailed validation with specific improvement recommendations.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=validation_prompt)
        result = chain.run(
            financial_text=financial_text,
            criteria="\n".join(f"- {c}" for c in self.validation_criteria)
        )
        
        return {
            "agent": "Notes",
            "analysis": result,
            "compliance_score": self._calculate_compliance_score(result),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_compliance_score(self, analysis_text: str) -> int:
        """Calculate compliance score"""
        positive_indicators = ["compliant", "adequate", "proper", "correct", "complete"]
        negative_indicators = ["missing", "incomplete", "non-compliant", "inadequate", "error"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_text.lower())
        negative_count = sum(1 for indicator in negative_indicators if indicator in analysis_text.lower())
        
        base_score = 50
        score = base_score + (positive_count * 10) - (negative_count * 15)
        return max(0, min(100, score))

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def main():
    st.title("🏦 Financial Report Validation System")
    st.markdown("### AI-Powered Compliance Validation with Meta Prompting")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable AI validation"
        )
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to proceed")
            return
        
        st.header("📋 Validation Scope")
        validate_balance_sheet = st.checkbox("Balance Sheet", value=True)
        validate_profit_loss = st.checkbox("Profit & Loss", value=True)
        validate_cash_flow = st.checkbox("Cash Flow", value=True)
        validate_notes = st.checkbox("Notes", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📄 Upload Financial Statement")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload the financial statement PDF for validation"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Extract text from PDF
            with st.spinner("Extracting text from PDF..."):
                financial_text = extract_text_from_pdf(uploaded_file)
            
            if financial_text:
                st.info(f"Extracted {len(financial_text)} characters from PDF")
                
                # Display text preview
                with st.expander("📖 Text Preview"):
                    st.text_area("Extracted Text", financial_text[:1000] + "...", height=200)
    
    with col2:
        st.header("🤖 AI Validation Results")
        
        if uploaded_file is not None and financial_text:
            
            if st.button("🚀 Start Validation", type="primary"):
                validation_results = []
                
                # Initialize agents
                agents = []
                if validate_balance_sheet:
                    agents.append(("Balance Sheet", BalanceSheetAgent(openai_api_key)))
                if validate_profit_loss:
                    agents.append(("P&L", ProfitLossAgent(openai_api_key)))
                if validate_cash_flow:
                    agents.append(("Cash Flow", CashFlowAgent(openai_api_key)))
                if validate_notes:
                    agents.append(("Notes", NotesAgent(openai_api_key)))
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (agent_name, agent) in enumerate(agents):
                    status_text.text(f"Validating {agent_name}...")
                    
                    try:
                        if agent_name == "Balance Sheet":
                            result = agent.validate_balance_sheet(financial_text)
                        elif agent_name == "P&L":
                            result = agent.validate_profit_loss(financial_text)
                        elif agent_name == "Cash Flow":
                            result = agent.validate_cash_flow(financial_text)
                        elif agent_name == "Notes":
                            result = agent.validate_notes(financial_text)
                        
                        validation_results.append(result)
                        
                    except Exception as e:
                        st.error(f"Error in {agent_name} validation: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(agents))
                
                status_text.text("Validation complete!")
                
                # Display results
                st.subheader("📊 Validation Dashboard")
                
                # Overall compliance score
                if validation_results:
                    avg_score = sum(r.get('compliance_score', 0) for r in validation_results) / len(validation_results)
                    
                    col_score1, col_score2, col_score3 = st.columns(3)
                    with col_score1:
                        st.metric("Overall Compliance", f"{avg_score:.1f}%")
                    with col_score2:
                        st.metric("Agents Used", len(validation_results))
                    with col_score3:
                        st.metric("Issues Found", sum(1 for r in validation_results if r.get('compliance_score', 100) < 80))
                
                # Individual agent results
                for result in validation_results:
                    with st.expander(f"🔍 {result['agent']} Analysis (Score: {result.get('compliance_score', 0):.1f}%)"):
                        
                        # Display refined analysis if available (for Balance Sheet)
                        if 'refined_analysis' in result:
                            st.subheader("Initial Analysis")
                            st.write(result.get('initial_analysis', 'N/A'))
                            st.subheader("Refined Analysis (After Self-Reflection)")
                            st.write(result.get('refined_analysis', 'N/A'))
                        else:
                            st.write(result.get('analysis', 'N/A'))
                        
                        # Compliance score visualization
                        score = result.get('compliance_score', 0)
                        if score >= 80:
                            st.success(f"✅ Good Compliance: {score:.1f}%")
                        elif score >= 60:
                            st.warning(f"⚠️ Moderate Compliance: {score:.1f}%")
                        else:
                            st.error(f"❌ Poor Compliance: {score:.1f}%")
                
                # Download results
                if validation_results:
                    results_json = json.dumps(validation_results, indent=2)
                    st.download_button(
                        label="📥 Download Validation Report",
                        data=results_json,
                        file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.info("Please upload a financial statement PDF to begin validation")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 Powered by AI Meta Prompting | 🔒 Secure | 📊 Accurate</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

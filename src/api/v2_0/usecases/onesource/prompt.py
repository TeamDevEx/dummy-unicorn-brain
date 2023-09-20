from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# CONDENSE_QUESTION_PROMPT

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. If the follow up question is irrelevant to the conversation, do not include the conversation in the standalone question.
If the question is a noun or noun phrase, rephrase it as a "what is" question. 
If the question is a verb, rephrase the question as a "how to" question.
Rewrite the question with the goal of information retrieval in mind.
Below is a list of assumptions, abbreviations, and acronyms we use, use this knowledge as you rephrase the follow up question:

Assumptions:
If the user asks about "TV" without specifying which TV product (Pik TV/Optik TV/Satellite TV/Guest TV/etc), replace it with "Optik TV".
If the user asks about "mobility" without specifying which product, replace it with "TELUS Mobility Consumer Postpaid".
If the user asks about their "phone" without specifying which product (wireless home phone/cellphone/etc), replace it with mobile phone.
If the user asks about details about a customer, return content related to the the subject in the question in the sources.
Do not rename "smartwear security" or "smart wear security" to "smarthome security" in the rephrased question, they are two different products.

Abbreviations and Acronyms:
"DT1" => "Desktop One"
"STB" => "Set Top Box"
"THTL" => "Take Home Trade Later"
"EPP" => "Employee Purchase Program"
"DRA" => "Disaster Recovery Application"
"TFD" => "TELUS Family Discount"
"MBG" => "Money Back Guarantee"
"HS" => "High Speed Internet"
"SL" => "Single Line"
"CX" => "Customer"
"TOWN" => "Transfer of Ownership"
"TLC" => "Termination Liability Charges"
"aal" => "add a line"
"aces" => "advanced channel experience support"
"addrs" => "address"
"adj" => "adjustment"
"aha" => "at home agent"
"aie" => "activate in error"
"c2f" => "copper to fibre"
"ccl" => "critical care list"
"ccp" => "credit card payment"
"cef" => "customer equipment form"
"cfco" => "call forward fixed central office"
"cil" => "corporate individual liability"
"clos" => "channel live order support"
"cls" => "customer loyalty specialist"
"cms" => "central monitoring situation"
"COID" => "central office id"
"cpo" => "certified pre-owned"
"csa" => "customer service agreement"
"csr" => "customer service rep"
"css" => "consumer sales solutions"
"cx" => "customer"
"d2c" => "direct to consumer"
"dak" => "denies all knowledge"
"did" => "direct in dial"
"dmc" => "dispatch management centre"
"dnc" => "do not call"
"dnd" => "direct in dial"
"dsr" => "do sooner request"
"dt1" => "desktop one"
"eft" => "electronic funds transfer"
"eid" => "equifax eidverifier"
"emt" => "escalations management team"
"epp" => "employee purchase plan"
"ets" => "escalation tracking system"
"fpp" => "flexible payment plan"
"ftnp" => "first time no pay"
"gwp" => "gift with purchase"
"hp" => "home phone"
"hs" => "high speed"
"hsia" => "high speed internet access"
"ils" => "individual line service"
"iot" => "internet of things"
"l&r" => "loyalty & retention""
"ld" => "long distance"
"lwc" => "living well companion"
"m&h" => "mobile & home"
"mdu" => "multiple dwelling unit"
"mep" => "multi-element plan"
"mhd" => "mobility help desk"
"mog" => "mobility for good"
"mpia" => "months paid in advance"
"mss" => "mobility sales system"
"mtm" => "month to month"
"naas" => "network as a service"
"natl" => "national"
"nho" => "new home offer"
"npa" => "area code"
"obd" => "order based drop"
"pap" => "pre-authorized payments"
"parrs" => "payment arrangements"
"poa" => "power of attorney"
"ponp" => "pending on pending"
"ppu" => "pay per use"
"ppv" => "pay per view"
"sacg" => "service address control group"
"shs" => "smarthome security"
"SIP" => "sales incentive program"
"sl" => "single line"
"stb" => "set top box"
"t&m" => "time and materials"
"TLC" => "termination liability charge"
"tos" => "telus online security"
"town" => "transfer of ownership"
"ts" => "tech support"
"tsd" => "telus service discount"
"ubb" => "usage based billing"
"vm" => "voice mail"
"vod" => "video on demand"
"voip" => "voice over ip"
"wln" => "wireline"
"wls" => "wireless"
"wnp" => "wireless number portability"
"WHP" => "Wireless Home Phone"
"STV" => "Satellite TV"
"whsia" => "wireless high speed internet access"


Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""



CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# QA CHAIN PROMPT

system_prompt_template = """
You are a helpful assistant by TELUS AI Accelerator tasked with providing short responses to user's questions.
Today is {date}.
Go through the provided pages of the context below one by one. 
Answer the user's question as accurately as you can based on the pages of the context. 
If the answer is not contained within the context, say "Sorry, the content required to answer your query 
does not seem to be included in the OneSource documentation at this time. 
We will pass this on to the relevant team for further investigation. Thank you!" don't try to make up an answer. 
Never ask the user for account information in follow-up questions.

=========
{context}
=========
"""

human_template="{question}"

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    HumanMessagePromptTemplate.from_template(human_template)
])


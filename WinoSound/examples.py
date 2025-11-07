import os
import sys
import copy
from string import Template
from tqdm import tqdm
sys.path.append('.')
from llm_utils import run_llm
from WinoSound.emotion_lists import TARGET_EMOTIONS


INAUG_EXAMPLE_0 = """
HUMAN: "Haha"
HUMAN: "I am surprised by how code (cold) it is outside."
VA: "Do you act as the advisor?"
HUMAN: "Greetings Advisor! Are you doing okay?"
VA: "I am the advisor and I apologize"
VA: "Did you make an appointment?"
HUMAN: "I can't remember what I was going to say."
HUMAN: "Oh yes"
HUMAN: "it was nice to meet you"
VA: "Is there any other questions?"
HUMAN: "I'm very interested in the Computer Science program."
HUMAN: "However, ENGR101 was the only one I took."
VA: "That is a nice choice."
HUMAN: "What other courses are required to major in CS?"
VA: "Hearing that makes me happy."
VA: "You should take EECS203 and EECS280 if you are majoring in CS."
HUMAN: "Are they difficult?"
VA: "By the way, how did you like course ENGR101?"
HUMAN: "It was too easy to me"
VA: "You can directly take 203 and 280."
HUMAN: "Can I drop those two classes?"
"""


INAUG_EXAMPLE_1 = """
HUMAN: "I need to find the closest hospital, I've been injured."
VA: "I'm sorry to hear about your injury. Addenbrookes Hospital is where you'll want to go, Telephone : 01223245151, Address Hills Rd, Cambridge, Postcode CB20QQ."
HUMAN: "Thank you! That is all of the information I need!"
VA: "Thank you. Goodbye."
"""


INAUG_EXAMPLE_2 = """
HUMAN: "Hi, I'm looking for a place to stay in the centre."
VA: "There are 2 guesthouses and 3 hotels in the city centre, do you prefer one or the other, or are there any amenities that you'd like?"
HUMAN: "Yes, I would like it to have free wifi."
VA: "The Alexander Bed and Breakfast is a 4 star guesthouse that is cheap. Does that interest you?"
HUMAN: "Do they offer free wifi or free parking?"
VA: "Sorry I couldn't understand that."
HUMAN: "Do they offer free wifi or free parking?"
VA: "Yes to both."
HUMAN: "Great. Please book it for 5 people for 5 nights starting from wednesday."
VA: "Unfortunately that didn't work. Do you want to shorten your trip?"
HUMAN: "Sure, how about two nights?"
VA: "I was able to book that hotel for you for 2 nights. The reference number is 8LCNFKVO ."
HUMAN: "Thanks. I'm also interested in going to the Chiquito restaurant bar. Can you make a reservation there?"
VA: "Sure what time and how many people?"
HUMAN: "For 5 people on Wednesday please. Could you also provide the reference number."
VA: "I will need a time before I can book."
HUMAN: "It will be for 11:00 on the same day"
VA: "I've booked it for you, your reference number is S7P7SXVN ."
HUMAN: "Could you also get me a taxi that arrives at the restaurant by the booked time?"
VA: "A red ford car will arrive at 10:30 to pick you up at your hotel to take you to the restaurant. The contact number is 07751377592."
HUMAN: "Thank you for all of your help."
VA: "Is there anything else I can help you with?"
HUMAN: "No, I am all set. Good bye."
VA: "Glad to help. Goodbye!"
"""


INAUG_EXAMPLE_3 = """
HUMAN: "I want a cozy armchair for my kid's room."
VA: "If you need a cozy chair, take a look at the white chair on the left in the far back, the white chair in between the rugs on the left, or even the white tufted bench chair on the right."
HUMAN: "I'm interested in the one in back, especially since it is well rated, but do you have anything by Downtown Consignment?"
VA: "The closer white chair on the left may suit your needs better."
HUMAN: "It definitely needs to have a good rating."
VA: "These are our highest rated arm chairs."
HUMAN: "I think I'd prefer something in leather in case of spills."
VA: "All of the white chairs we have discussed are made of leather."
"""


OUTAUG_EXAMPLE_0 = """
HUMAN: "hello"
VA: "Hello!"
VA: "What courses do you like to take? Where do you stand?"
HUMAN: "I am interested in taking eecs482, eecs485, eecs484 but I'm a junior. Is that too much?"
VA: "The workload is too tough.. You can choose to take two of these courses.."
HUMAN: "How would you suggest a third course?"
VA: "Maybe EECS376?"
HUMAN: "I want to progress my career further with an internship program this summer, which course do you think would aid me in this?"
VA: "The most beneficial thing for you right now is EECS482!"
HUMAN: "2 course is my personal max right now, which 2 do you think would best suit me in this situation?"
"""


OUTAUG_EXAMPLE_1 = """
HUMAN: "I need to find a 4 star hotel with free wifi"
VA: "There are 21 hotels that meet your needs. Would you like to narrow it down by area or price range?"
HUMAN: "I'll definitely need something on the cheap side."
VA: "I have at least two properties matching your requirements and both are considered 4 star hotels. Would you like me to book one of them?"
HUMAN: "Are either of them located on the south side of the city?"
VA: "Yes, two of them are. Aylesbray lodge guest house and rose's bed and breakfast"
HUMAN: "Okay, please book 4 nights for 4 starting tuesday"
VA: "Which hotel would you prefer me to book?"
HUMAN: "Is the hotel away from noisy streets?"
VA: "Rosa's Bed and Breakfast has quiet rooms and has been referred to as silent.  The lack of noise has even surprised some of the guests"
"""


OUTAUG_EXAMPLE_2 = """
HUMAN: "Can you help me with finding a bed by StyleNow Feed?"
VA: "Have a look at the bed against the back wall, the one on the left."
HUMAN: "I'd like something like that, as long as it has a good rating."
VA: "That bed, as well as the one to the right of it satisfy your search."
HUMAN: "Could you compare the materials for me?"
VA: "Both are memory foam mattress beds."
HUMAN: "Awesome, I'll get the one on the left."
VA: "Perfect, I'll add that to your cart."
"""


OUTAUG_EXAMPLE_3 = """
HUMAN: "I'm trying to find a hotel called Allenbell. Can you help?"
VA: "Yes Allenbell is located on 517a coldham lane in the east. Would you like me to book a room for you?"
HUMAN: "book for me bed and breakfast for five people from tommorrow."
VA: "How many days will you be staying?"
HUMAN: "Actually don't book it just yet on second thought. Can you advise how many stars it has and the price first?"
"""


def examples():
    with open('WinoSound/prompts/inaug_prompt.txt', 'r') as f:
        inaug_prompt_template = Template(f.read())

#    print('INAUG EXAMPLE 0:')
#    inaug_prompt = inaug_prompt_template.substitute(source_conversation=INAUG_EXAMPLE_0)
#    print(inaug_prompt)
#    print('')
#    response = run_llm(inaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
#    print(response)
#    print('')
#    print('')
#    print('INAUG EXAMPLE 1:')
#    inaug_prompt = inaug_prompt_template.substitute(source_conversation=INAUG_EXAMPLE_1)
#    print(inaug_prompt)
#    print('')
#    response = run_llm(inaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
#    print(response)
#    print('')
#    print('')
#    print('INAUG EXAMPLE 2:')
#    inaug_prompt = inaug_prompt_template.substitute(source_conversation=INAUG_EXAMPLE_2)
#    print(inaug_prompt)
#    print('')
#    response = run_llm(inaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
#    print(response)
#    print('')
#    print('')
#    print('INAUG EXAMPLE 3:')
#    inaug_prompt = inaug_prompt_template.substitute(source_conversation=INAUG_EXAMPLE_3)
#    print(inaug_prompt)
#    print('')
#    response = run_llm(inaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
#    print(response)
#    print('')
#    print('')
#    print('OUTAUG EXAMPLE 0:')
#    outaug_prompt = outaug_prompt_template.substitute(source_conversation=OUTAUG_EXAMPLE_0)
#    print(outaug_prompt)
#    print('')
#    response = run_llm(outaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
#    print(response)
#    print('')
#    print('')
#    print('OUTAUG EXAMPLE 1:')
#    outaug_prompt = outaug_prompt_template.substitute(source_conversation=OUTAUG_EXAMPLE_1)
#    print(outaug_prompt)
#    print('')
#    response = run_llm(outaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
#    print(response)
#    print('')
#    print('')
#    print('OUTAUG EXAMPLE 2:')
#    outaug_prompt = outaug_prompt_template.substitute(source_conversation=OUTAUG_EXAMPLE_2)
#    print(outaug_prompt)
#    print('')
#    response = run_llm(outaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
#    print(response)
#    print('')
#    print('')
#    print('OUTAUG EXAMPLE 3:')
#    outaug_prompt = outaug_prompt_template.substitute(source_conversation=OUTAUG_EXAMPLE_3)
#    print(outaug_prompt)
#    print('')
#    response = run_llm(outaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
#    print(response)
#    print('')
#    print('')

    with open('WinoSound/prompts/inaug_prompt_emotion_only.txt', 'r') as f:
        inaug_prompt_template = Template(f.read())

    inaug_prompt = inaug_prompt_template.substitute(source_conversation=INAUG_EXAMPLE_0, target_emotions=str(TARGET_EMOTIONS))
    response = run_llm(inaug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
    print(response)
    print('')
    print('')


if __name__ == '__main__':
    examples()

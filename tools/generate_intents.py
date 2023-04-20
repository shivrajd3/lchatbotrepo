import json
from intent_generator import IntentsGenerator

full_file_name = '../../lchatbot/bot/static/loyalist_courses.json'
manual_intent_json_file = '../../lchatbot/bot/static/manual Intent - loyalist.json'

intentgen = IntentsGenerator()
program_attributes = intentgen.get_program_attributes()
question_types = intentgen.get_question_types()

# reading manual intent json file
manual_intent = json.loads(open(manual_intent_json_file).read())

new_manual_intent = {}
new_intent_response_dict = {}
intent_dict = {}
intents_list = []

for intent in manual_intent['intents']:
    
    intent_dict['tag'] = intent['tag']
    intent_dict['patterns'] = intent['patterns']
    intent_responses_list = intent['responses']
    response_list = []
    for intent_response in intent_responses_list:
        # print(intent_response)
        new_intent_response_dict['resp'] = intent_response
        response_list.append(new_intent_response_dict)
        new_intent_response_dict = {}
    intent_dict['responses'] = response_list
    intents_list.append(intent_dict)
    intent_dict = {}
    # print(intents_list)
    # input('testing')

new_manual_intent['intents'] = intents_list

# for intent in manual_intent['intents']:
#     intent_responses = intent['responses']
#     responses = []
#     for intent_response in intent_responses:
#         print(intent_response)
#         new_intent_response_dict['resp'] = intent_response
#         responses.append(new_intent_response_dict)
#         print('testing')
    
#     # print(responses)
#     intent_dict['tag'] = intent['tag']
#     intent_dict['patterns'] = intent['patterns']
#     intent_dict['responses'] = responses
#     # print(intent_dict)
#     intents.append(intent_dict)
#     print(intents)
#     input('testing')

# new_manual_intent = {'intents': intents}
# print(new_manual_intent)
# exit()
# manual_intent = new_manual_intent
# # print(manual_intent['intents'][1])
# # input('testing 1')

# print(manual_intent)


with open(full_file_name) as ipfile:
    data_lines = ipfile.readlines()

    courses_done = []
    for idx, data_line in enumerate(data_lines):
        print(idx)
        json_data = json.loads(data_line)
        # print(json_data)
        if json_data['course_title'] in courses_done:
            continue

        courses_done.append(json_data['course_title'])

        intent_dict = intentgen.generate_intent(
            attribute_name='course_title', course_json_data=json_data)

        # print(intent_dict)
        # input('testing')
        new_manual_intent['intents'].append(intent_dict)
        # if idx > 15:
        #     break

json_obj = json.dumps(new_manual_intent, indent=4)

with open('../../lchatbot/bot/static/Intent - loyalist.json', 'w') as opfile:
    opfile.write(json_obj)

print(f'saving file: ../../lchatbot/bot/static/Intent - loyalist.json')

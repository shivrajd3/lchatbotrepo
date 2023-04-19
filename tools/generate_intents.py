import json
from intent_generator import IntentsGenerator

full_file_name = '../lchatbot/bot/static/loyalist_courses.json'
manual_intent_json_file = '../lchatbot/bot/static/manual Intent - loyalist.json'

intentgen = IntentsGenerator()
program_attributes = intentgen.get_program_attributes()
question_types = intentgen.get_question_types()

# reading manual intent json file
manual_intent = json.loads(open(manual_intent_json_file).read())
# print(manual_intent['intents'][1])
# input('testing 1')

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
        manual_intent['intents'].append(intent_dict)
        # if idx > 15:
        #     break

json_obj = json.dumps(manual_intent, indent=4)

with open('../lchatbot/bot/static/Intent - loyalist.json', 'w') as opfile:
    opfile.write(json_obj)

print(f'saving file: ../lchatbot/bot/static/Intent - loyalist.json')

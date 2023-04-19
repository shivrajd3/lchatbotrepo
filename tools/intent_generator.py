import json

class IntentsGenerator:

    #   {'_id': '14b07a8cc5e54d9b8790b8f8eb62e4f6',
    #   'course_title': 'Advanced Filmmaking - Digital Content Creation ',
    #   'college_name': 'Loyalist', 'campus': 'Main', 'program_delivery': 'Full Time',
    #   'intake': 'Sep 2023',
    #   'program_status': 'Open',
    #   'program_url': 'https://www.loyalistcollege.com/programs-and-courses/full-time-programs/advanced-filmmaking-digital-content-creation/',
    #   'program_info':
    #       {'Program Length:': '1 Academic Years (Periods Of 8 Months)',
    #       'Program Code:': 'AFGS',
    #       'Credential:': 'Graduate Certificate',
    #       'Program Level:': 'Post-Diploma',
    #       'Program Type:': 'Regular',
    #       'Language:': 'English',
    #       'Entry Level:': 'Semester 1',
    #       'Highly Competitive:': 'No',
    #       'Program Status:': 'Open'}}

    def __init__(self):
        pass

    def get_program_attributes(self):
        return [
            'course_title',
            'college_name',
            'campus',
            'program_delivery',
            'intake',
            'program_status',
            'program_url',
            'program_info'
        ]

    def get_program_info_attributes(self):
        return [
            'Program Length:',
            'Program Code:',
            'Credential:',
            'Program Level:',
            'Program Type:',
            'Language:',
            'Entry Level:',
            'Highly Competitive:',
            'Program Status:'
        ]

    def get_question_types(self):
        ques_type_dict = {
            'course_title': [
                'tell me more about',
                'is there a course like',
                'is there a program like',
                'i would like to know about',
                'does this college teach',
                'do you teach',
                'i would like to learn'
            ]
        }
        return ques_type_dict

    def get_response_types(self):
        response_type_dict = {
            'course_title': [
                'here you go',
                'i guess here is what you need',
                'does this help?'
            ]
        }
        return response_type_dict

    def generate_intent(self, attribute_name, course_json_data):
        intent_dict = dict()
        patterns = list()

        program_attributes = self.get_program_attributes()
        program_info_attributes = self.get_program_info_attributes()

        question_types = self.get_question_types()
        response_types = self.get_response_types()
        program_attributes.remove('program_info')
        program_info = course_json_data['program_info']
        # print(f'program info: {program_info}')

        if attribute_name == 'course_title':
            intent_dict['tag'] = attribute_name + " " + \
                course_json_data[attribute_name]

            intent_dict['patterns'] = list()
            question_types = question_types[attribute_name]
            for question_type in question_types:
                # ques = ""
                # ques_parts = question_type.split(' ')
                # for ques_part in ques_parts:
                #     ques += ques_part + " "
                intent_dict['patterns'].append(question_type+" "+course_json_data[attribute_name])
                course_name_parts = course_json_data[attribute_name].split(' ')
                crs_name = ""
                for course_name_part in course_name_parts:
                    crs_name += course_name_part + " "
                    intent_dict['patterns'].append(course_name_part)
                    intent_dict['patterns'].append(crs_name)
                    intent_dict['patterns'].append(question_type+" "+crs_name)
                    intent_dict['patterns'].append(question_type+" "+course_name_part)

            program_attributes.remove(attribute_name)
            remaining_attributes = program_attributes
            # print(f'remaining_attributes: {remaining_attributes}')
            responses_list = list()
            for response_type in response_types[attribute_name]:
                response_body = dict()
                response_body['message'] = response_type
                response_body['course_details'] = dict()
                # response_string += response_type + " >> "
                # print(response_string)
                course_title_dict = {
                    'course_title': course_json_data[attribute_name]
                }
                response_body['course_details'].update(course_title_dict)
                for attr in remaining_attributes:
                    # print(f'adding: {attr}')
                    # response_string += attr+":"+course_json_data[attr] + " | "

                    attr_dict = {
                        attr: course_json_data[attr]
                    }
                    response_body['course_details'].update(attr_dict)
                    # print(response_string)
                    # pass
                response_body['program_info'] = program_info
                # response_string += "Program Info >> "+ json.dumps(program_info)
                # for prog_info_attr in program_info_attributes:
                #     response_string += prog_info_attr + \
                #         program_info[prog_info_attr] + " | "

                responses_list.append({'response': response_body})

            intent_dict['responses'] = responses_list

        return intent_dict

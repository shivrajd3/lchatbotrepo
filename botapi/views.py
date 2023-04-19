from bot.views import generate_response
from django.shortcuts import render
from rest_framework import viewsets
from .serializers import BotResponseSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
import json
import logging
logger = logging.getLogger('django')

# Create your views here.


class ResponseClass:
    def __init__(self, botresponse):
        self.botresponse = botresponse


class askthebot(viewsets.ViewSet):
    # permission_classes = (IsAuthenticated,)

    # if request.method == 'POST':
    def create(self, request):
        user_msg = request.data
        user_msg_data = user_msg['user_msg']
        # botresponse = generate_response("hi there")
        botresponse = generate_response(user_msg_data)
        botresponse = botresponse.content
        botresponse_json = json.loads(botresponse)

        # botresponse = botresponse['response']
        logger.info(f'botresponse received: {botresponse}')
        logger.info(f'botresponse json: {botresponse_json}')
        logger.info(f"user_message: {user_msg['user_msg']}")
        logger.info(
            f"botresponse json response: {botresponse_json['response']}")
        # botresponseobj = ResponseClass(botresponse=botresponse_json)
        botresponseobj = ResponseClass(botresponse=botresponse_json['response'])
        serializer = BotResponseSerializer(botresponseobj)
        logger.info(f"serializer data bot response: {serializer.data['botresponse']}")
        return Response(serializer.data['botresponse'])

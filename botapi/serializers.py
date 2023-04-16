from rest_framework import serializers


class BotResponseSerializer(serializers.Serializer):
    botresponse = serializers.CharField(max_length=600)

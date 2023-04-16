from django.urls import path
from . import views

urlpatterns = [
    # frontend URLs
    path('train', views.train_model, name='train_model'),
    path('generate_response', views.generate_response, name='generate_response'),

    # path('', views.index, name='contenthome'),
    # path('login', views.login_view, name='login'),
    # path('logout', views.logout_view, name='logout'),
    # path('signup', views.signup, name='signup'),
    # path('writecontent', views.writecontent, name='writecontent'),
    # path('manageproductattributes', views.manageproductattributes, name='manageproductattributes'),
    # path('searchproduct', views.searchproduct, name='searchproduct'),
    # path('editattributes/<str:sku>', views.editattributes, name='editattributes'),
    # path('editcontent', views.editcontent, name='editcontent'),
    # path('editcontent/<str:listing_id>', views.editcontent, name='editcontent'),
    # path('validate_images/<int:subcategory_id>', views.validate_images, name='validate_images'),
    # path('getvariations/<str:listing_id>', views.getvariations, name='getvariations'),

    # # tools URLs
    # path('import_mongo_to_orm', views.import_mongo_to_orm, name='import_mongo_to_orm'),
]

import qrcode
q=qrcode.QRCode(version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=50,
                border=1)
q.add_data("BELEIVE IN YOU")
q.make(fit=True)
img=q.make_image(fill_color="black",back_color="white")
img.save("C:\\Users\\sai venkat dhulipudi\\Desktop/Y U STARTED.jpg")


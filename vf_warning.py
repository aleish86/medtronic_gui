import easygui

def vf_warning_msg():
    easygui.msgbox(msg="Warning: VF Detected", title="VF WARNING!", image = "warning_sign.jpeg", ok_button="Acknowledged")


if __name__ == "__main__":
    vf_warning_msg()

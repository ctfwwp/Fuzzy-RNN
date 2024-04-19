def check(modbus_tcp):
    fc = modbus_tcp[14:16]
    flag = True
    reasion=''
    if fc =="01" or fc == "02" or fc =="03" or fc == "04" or fc == "06":
        if len(modbus_tcp)!=24:
            flag = False

    elif fc == "05":
        if len(modbus_tcp) != 24:
            flag = False
        elif len(modbus_tcp) == 24:
            if modbus_tcp[-4:]!="FF00" and modbus_tcp[-4:]!="ff00" and modbus_tcp[-4:]!="0000":
                flag = False

    elif fc =="10":
        if len(modbus_tcp) <= 26:
            flag = False
        elif int(modbus_tcp[24:26], 16)*2 != len(modbus_tcp[26:]):
            flag = False
        elif int(modbus_tcp[20:24], 16)*2 != int(modbus_tcp[24:26], 16):
            flag = False

    elif fc == "0F" or fc == "0f":
        if len(modbus_tcp) < 28:
            flag = False
        elif int(modbus_tcp[24:26], 16) * 2 != len(modbus_tcp[26:]):
            flag = False
    else:
        pass
    return flag,reasion


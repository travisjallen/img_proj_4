def round(test_int):
    str_int = str(test_int)
    last_digit = int(str_int[-1])
    if(last_digit >= 5):
        str_int.replace(str_int[-1],'0')
        str_int.replace(str_int[-2],str(int(str_int[-2]) + 1))
    else:
        str_int.replace(str_int[-1],'0')
    return int(str_int)


test_int = 35
re = round(test_int)
print(re)
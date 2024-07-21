def friendly_num(num: int) -> str:
    if num > 1e9:
        return f"{num/1e9:03.2f}B"
    elif num > 1e6:
        return f"{num/1e6:03.2f}M"
    elif num > 1e3:
        return f"{num/1e3:03.2f}K"
    else:
        return f"{num}"
    
if __name__ == "__main__":
    print(friendly_num(12))
    print(friendly_num(1234))
    print(friendly_num(12345))
    print(friendly_num(123456))
    print(friendly_num(1234567))
    print(friendly_num(12345678))
    print(friendly_num(123456789))
    print(friendly_num(1234567890))
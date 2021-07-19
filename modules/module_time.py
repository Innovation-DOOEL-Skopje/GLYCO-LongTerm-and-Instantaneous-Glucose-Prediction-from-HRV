from datetime import datetime


def hmymd():
    """Get datetime string in %Hh%Mm-%Y%m%d format"""
    return datetime.now().strftime("%Hh%Mm-%Y%m%d")

# ____________________________________________________________________________________________________________________________________


def dm():
    """Get datetime string in %m.%d format"""
    return datetime.now().strftime("%d.%m")

# ____________________________________________________________________________________________________________________________________


def ymd():
    """Get datetime string in %Y%m%d format"""
    return datetime.now().strftime("%Y%m%d")




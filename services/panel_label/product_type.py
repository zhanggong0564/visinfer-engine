'''
@Author       : gongzhang4
@Date         : 2026-02-28 07:32:47
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 11:13:46
@FilePath     : product_type.py
@Description  :
'''

import json

PRODUCT_TYPE = {
    "QF2": [
        "QF2-1/PE1-J1",
        "QF2-3/PE1-J3",
        "QF2-5/PE1-J5",
        "FU34-2/KM1-1",
        "FU35-2/KM1-3",
        "FU36-2/KM1-5",
        "QF2-2/T1-38V-a",
        "QF2-4/T1-38V-b",
        "QF2-6/T1-38V-c",
        "FU34-1/QS2-OUT+2",
        "FU35-2/QS2-OUT-3",
        "FU36-2/QS2-OUT+3",
    ],
    "PE1-A": ["PE1-J6/QF2-5", "PE1-J5/QF2-5", "PE1-J4/QF2-3", "PE1-J3/QF2-3", "PE1-J2/QF2-1", "PE1-J1/QF2-1"],
    "PE1-B": [
        "PE1-J12/PD-J54-1",
        "PE1-J12/FAN6--",
        "PE1-J11/FAN2--",
        "PE1-J10/FAN1--",
        "PE1-J9/FAN6-+",
        "PE1-J8/FAN2-+",
        "PE1-J7/FAN1-+",
    ],
    "T1": [
        "T1-38V-c/QF2-6",
        "T1-38V-b/QF2-4",
        "T1-38V-a/QF2-2",
        "T1-230V/FU37-1",
        "T1-n2/PD-J9-2",
        "T1-390V/FU38-1",
        "T1-n1/PE-J6-3",
        "T1-C2-660V/KM1-6",
        "T1-B2-660V/KM1-4",
        "T1-A2-660V/KM1-2",
    ],
    "PH": ["PH-J1-1/PD-J3-2", "PH-J1-3/PD-J3-5", "PH-J2/PD-J32"],
    "S1S2": ["S2-14/PD-J22-1", "S2-13/PD-J22-2", "S1-13/PD-J27-2", "S1-14/PD-J27-1"],
    "D1": ["D1-1/R2-2", "D1+/DC+", "D1-/DC-", "D1-3/R3-2"],
    "QF1L1": ["QF1-1/C2-L1", "QF1-1/C4-L1", "QF1-1/C1-L1", "QF1-1/C3-L1"],
    "QF1L2": ["QF1-3/C2-L2", "QF1-3/C4-L2", "QF1-3/C1-L2", "QF1-3/C3-L2"],
    "QF1L3": ["QF1-5/C4-L3", "QF1-5/C2-L3", "QF1-5/C3-L3", "QF1-5/C1-L3"],
    "XB3": [
        "FU31-1/DC+",
        "FU32-1/DC-",
        "FU37-1/T1-230V",
        "FU38-1/T1-390V",
        "FU29-1/KM3-1",
        "FU30-1/KM3-3",
        "XB3-1A/PE24V",
        "XB3-2A/PD-J1-8",
        "XB3-5A/PE-24VGND",
        "XB3-6A/PD0J1-4",
        "FU31-2/PD-J11-8",
        "FU32-2/PD-J11-1",
        "FU37-2/PD-J9-3",
        "FU38-2/PE-J6-1",
        "FU29-2/QF1-6",
        "FU30-2/QF1-4",
    ],
    "J28J30": [
        "PD-J28-1/PN1-J5-1",
        "PD-J28-3/PN1-J5-3",
        "PD-J30-1/L-4",
        "PD-J30-3/L-3",
    ],
    "J3": [
        "PD-J3-6/PWI-J1-3",
        "PD-J3-5/PH-J1-3",
        "PD-J3-3/PWI-J1-4",
        "PD-J3-2/PH-J1-1",
        "PD-J3-1/KMI-A2",
    ],
    "J46": [
        "PD-J46-1/PW1-J2-2",
        "PD-J46-2/PW1-J2-1",
        "PD-J46-4/PW1-J2-5",
        "PD-J46-5/PW1-J2-4",
    ],
}

with open("product_type.json", "w") as f:
    json.dump(PRODUCT_TYPE, f, ensure_ascii=False, indent=4)


PRODUCT_guideline = {
    "QF2": [48, 467, 5148, 7174],
}

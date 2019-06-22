# everything init with zero
import numpy as np
MAXN = 10
PLEN = 64
SLEN = 8
MAX_TOT_LINE = 10000


class PTableType:
    def __init__(self):
        self.input_size = 0
        self.output_size = 0
        self.element = np.zeros(PLEN)


class SBoxType:  # SBox
    def __init__(self):
        self.input_size = 0
        self.output_size = 0
        self.element = np.zeros(pow(2, SLEN))


class LexType:  # 保留字符和符号的解析
    def __init__(self):
        self.name = ""
        self.type = 0
        self.value = 0


class BSVarType:  # 分组语句
    def __init__(self):
        self.name = ""
        self.len = 0
        self.value = 0
        self.ch = 0


class ExprType:  # 表达式
    def __init__(self):
        self.id = 0
        self.type = 0
        self.value = 0
        self.bs_varp = []
        self.expr1 = 0
        self.expr2 = 0


class GraType:
    def __init__(self):
        self.type = 0
        self.var = 0
        self.value = 0
        self.jmp = 0
        self.bs_varp = []
        self.expr = 0
        self.st = 0
        self.ed = 0


class CIPHER:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.PTable = [PTableType() for i in range(MAXN)]
        self.SBox = [SBoxType() for i in range(MAXN)]
        self.TaskId = ""
        self.code = list()
        self.LoopVar = np.zeros(26)
        self.BsVar = []
        self.Lex = []
        self.gra = []
        self.LoopPos = []
        self.ExprPtr = []

    def read_ptable(self,):
        for i in range(self.n):
            ab = input("请输入置换表输入和输出的二进制串长度:")
            self.PTable[i].input_size = int(ab.split(" ")[0])
            self.PTable[i].output_size = int(ab.split(" ")[1])
            abcd = input("请输入b个用空格分隔的整数:")
            self.PTable[i].element = [int(j) for j in abcd.split(" ")]

    def read_sbox(self,):
        for i in range(self.m):
            cd = input("请输入该S盒对应输入、输出二进制串的长度:")
            self.SBox[i].input_size = int(cd.split(" ")[0])
            self.SBox[i].output_size = int(cd.split(" ")[1])
            cdef = input("请输入d^2个用空格分隔的整数:")
            self.SBox[i].element = [int(j) for j in cdef.split(" ")]

    def read_code(self,):
        while True:
            code_line = input()
            self.code += code_line + "\n"
            if code_line == "END":
                break

    def find_bs(self, name):
        for i in range(len(self.BsVar)):
            if self.BsVar[i].name == name:
                return i
        return -1

    def lex_ana(self,):  # 对代码进行解析
        type5 = ['=', '+', '(', ')', '[', ']', '\n']
        type0 = ["BEGIN", "END", "P", "S", "LOOP", "ENDLOOP", "SPLIT", "MERGE"]
        j = 0
        i = 0
        str1 = ""
        while i < len(self.code):
            while self.code[i] == ',' or self.code[i] == ' ' or self.code[i] == '\t':
                i += 1
            str1 += self.code[i]
            i += 1
            if str1[0].isupper():  # 如果是大写字母
                while i < len(self.code) and self.code[i].isupper():
                    str1 += self.code[i]
                    i += 1
                while j < len(type0):
                    if str1 == j:
                        break
                    j += 1
                temp1 = LexType()
                temp1.name = str1
                temp1.type = 0
                temp1.value = j
                self.Lex.append(temp1)
            elif str1[0].islower():  # 如果是小写字母
                while i < len(self.code) and self.code[i].islower():
                    str1 += self.code[i]
                    i += 1
                if len(str1) == 1:   # type
                    temp2 = LexType()
                    temp2.name = str1
                    temp2.type = 1
                    temp2.value = ord(str1[0]) - ord('a')
                    self.Lex.append(temp2)
                else:
                    j = self.find_bs(str1)
                    if j == -1:
                        j = len(self.BsVar)
                        Beta = BSVarType()
                        Beta.name = str1
                        Beta.len = 0
                        Beta.value = 0
                        self.BsVar.append(Beta)
                    temp3 = LexType()
                    temp3.name = str1
                    temp3.type = 2
                    temp3.value = j
                    self.Lex.append(temp3)
            elif str1.isdigit():  # 如果是数字
                while i < len(self.code) and self.code[i].isdigit():
                    str1 += self.code[i]
                    i += 1
                temp4 = LexType()
                temp4.name = str1
                temp4.type = 3
                temp4.value = ord(str1[0]) - ord('0')
                while j < len(str1):
                    temp4.value = temp4.value * 10 + ord(str1[j]) - ord('0')
                    j += 1
                self.Lex.append(temp4)
            elif str1[0] == "\\":
                while ord('0') <= ord(self.code[i]) <= ord('1'):
                    str1 += self.code[i]
                    i += 1
                str1 += self.code[i]
                i += 1
                temp5 = LexType()
                temp5.name = str1
                temp5.type = 4
                temp5.value = 0
                while j < len(str1) - 1:
                    temp5.value = (temp5.value << 1) + ord(str1[j]) - ord('0')
                self.Lex.append(temp5)
            else:
                while j < 7:
                    if str1[0] == type5[j]:
                        break
                    j += 1
                temp6 = LexType()
                temp6.name = str1
                temp6.type = 5
                temp6.value = j
                self.Lex.append(temp6)
            str1 = ""

    def init_bs_var(self,):
        i = 0
        while self.Lex[i].name != "BEGIN":
            temp = self.Lex[i].value
            beta = int(self.Lex[i+2].name)
            self.BsVar[temp].len = beta
            self.BsVar[temp].value = 0
            i += 5

    def gra_ana(self,):  # 对加密代码进行解析
        for i in range(len(self.Lex)):
            if self.Lex[i].name == "BEGIN":
                while i < len(self.Lex):
                    for j in range(i, len(self.Lex)):
                        if self.Lex[j].name == "\n":
                            self.generate_gra(i, j)  # 对每一行生成表达式
                            break
                    i = j + 1
                break

    def generate_gra(self, l, r):
        tmp = GraType()
        tmp.st = l
        tmp.ed = r
        tmp.var = tmp.value = tmp.jmp = 0
        tmp.expr = None
        if self.Lex[l].name == "BEGIN":
            tmp.type = 0
            tmp.value = 0
        elif self.Lex[l].name == "END":
            tmp.type = 0
            tmp.value = 1
        elif self.Lex[l].name == "LOOP":
            tmp.type = 2
            tmp.var = self.Lex[l + 1].value
            tmp.value = self.Lex[l + 2].value
            tmp.jmp = self.Lex[l + 3].value
            self.LoopPos.append(len(self.gra))
        elif self.Lex[l].name == "ENDLOOP":
            pos = self.LoopPos.pop()
            tmp.type = 3
            tmp.var = self.gra[pos].var
            tmp.value = self.gra[pos].jmp
            self.gra[pos].jmp = 0
            tmp.jmp = pos + 1
        elif self.Lex[l].name == "SPLIT":
            tmp.type = 4
            tmp.value = self.Lex[r - 2].value
            beta = 0
            _, __, beta, tmp.bs_varp = self.get_bs_var(l + 2, r - 3, beta, tmp.bs_varp)
            tmp.var = beta
        elif self.Lex[l].name == "MERGE":
            tmp.type = 5
            beta = 0
            _, __, beta, tmp.bs_varp = self.get_bs_var(l + 2, r - 2, beta, tmp.bs_varp)
            tmp.var = beta
        elif self.Lex[l].type == 2:
            i = l
            while i <= r and self.Lex[i].name != "=":
                i += 1
            tmp.type = 1
            beta = 0
            self.get_bs_var(l, i - 1, beta, tmp.bs_varp)
            tmp.var = beta
            tmp.expr = self.generate_expr(i + 1, r - 1)
        self.gra.append(tmp)

    def get_bs_var(self, l, r, value, bs_varp):
        value = self.Lex[l].value
        i = l + 1
        while i < r:
            if self.Lex[i + 1].type == 1:
                bs_varp.append(int(self.Lex[i + 1].value) - 26)
            else:
                bs_varp.append(self.Lex[i + 1].value)
            i += 3
        return l, r, value, bs_varp

    def generate_expr(self, l, r):
        ptr = ExprType()
        ptr.id = len(self.ExprPtr)
        self.ExprPtr.append(ptr)
        ptr.expr1 = None
        ptr.expr2 = None
        ptr.value = 0
        num = 0
        i = l
        while i <= r:
            if self.Lex[i].name == "(":
                num += 1

            if self.Lex[i].name == ")":
                num -= 1
            if self.Lex[i].name == "+" and num == 0:
                break
            i += 1
        if i <= r:
            ptr.type = 2
            ptr.expr1 = self.generate_expr(l, i - 1)
            ptr.expr2 = self.generate_expr(i + 1, r)
            return ptr
        if self.Lex[l].name == "P" or self.Lex[l].name == "S":
            if self.Lex[l].name == "P":
                ptr.type = 3
            else:
                ptr.type = 4
            if self.Lex[l + 2].type == 1:
                ptr.value = self.Lex[l+2].value - 26
            elif self.Lex[l+2].type == 3:
                ptr.value = self.Lex[l+2].value
            ptr.expr1 = self.generate_expr(l + 5, r - 1)
            return ptr
        if self.Lex[l].type == 4:
            ptr.type = 1
            ptr.value = self.Lex[l].value
            return ptr
        if self.Lex[l].type == 2:
            ptr.type = 0
            self.get_bs_var(l, r, ptr.value, ptr.bs_varp)
            return ptr

    def solve_task1(self,):
        k = input()
        for i in range(int(k)):
            str1 = self.bs_to_num(input())
            str2 = self.bs_to_num(input())
            a = self.encrypt(str1, str2)
            print(self.num_to_bs(a, "", self.BsVar[0].len))

    def find_bs_varp(self, pos, bs_varp):
        ptr = self.BsVar[pos]
        for i in range(len(bs_varp)):
            j = bs_varp[i]
            if j < 0:
                j = self.LoopVar[j + 26]
            ptr = ptr.ch[j]
        return ptr

    def cal_expr(self, ptr):  # 根据表达式类型计算出表达式结果
        if ptr is None:
            return 10
        if ptr.type == 0:  # 找变量中的地址
            return self.find_bs_varp(ptr.value, ptr.bs_varp).value
        elif ptr.type == 1:  # 直接赋值
            return ptr.value
        elif ptr.type == 2:
            return self.cal_expr(ptr.expr1) ^ self.cal_expr(ptr.expr2)
        elif ptr.type == 3:    # 带有置换表的
            tmp = self.cal_expr(ptr.expr1)
            j = ptr.value
            if j < 0:
                j = self.LoopVar[j+26]
            sout = ""
            sin = self.num_to_bs(tmp, "", self.PTable[j].input_size)
            for i in range(self.PTable[j].output_size):
                sout += sin[self.PTable[j].element[i]]
            tmp = self.bs_to_num(sout)
            return tmp
        elif ptr.type == 4:  # 带有SBOX的表达式处理
            tmp = self.cal_expr(ptr.expr1)
            j = ptr.value
            if j < 0:
                j = self.LoopVar[j+26]
            return self.SBox[j].element[int(tmp)]

    def bs_to_num(self, string):
        num = 0
        for i in range(len(string)):
            num = (num << 1) + ord(string[i]) - ord('0')
        return num

    def num_to_bs(self, num, string, size):
        string = ""
        tmp = 1 << (size - 1)
        while tmp:
            if num & tmp:
                string += '1'
            else:
                string += '0'
            tmp = tmp >> 1
        return string

    def encrypt(self, state, key):
        now = 0
        while True:
            if self.gra[now].type == 0:  # BEGIN 保留字
                if self.gra[now].value == 0:
                    self.BsVar[0].value = state
                    self.BsVar[1].value = key
                    for i in range(2, len(self.BsVar)):
                        self.BsVar[i].value = 0
                    now += 1
                else:  # END
                    # print(self.BsVar[0].value)
                    return self.BsVar[0].value
            elif self.gra[now].type == 1:  # 如果只是普通变量或者表达式
                ptr = self.find_bs_varp(self.gra[now].var, self.gra[now].bs_varp)
                ptr.value = self.cal_expr(self.gra[now].expr)
                now += 1

            elif self.gra[now].type == 2:  # 初始化循环控制变量
                self.LoopVar[self.gra[now].var] = self.gra[now].value
                now += 1
            elif self.gra[now].type == 3:  # 循环控制变量+1
                self.LoopVar[self.gra[now].var] += 1
                if self.LoopVar[self.gra[now].var] > self.gra[now].value:
                    now += 1
                else:
                    now = self.gra[now].jmp
            elif self.gra[now].type == 4:  # 分组  对明文进行拆分
                ptr = self.find_bs_varp(self.gra[now].var, self.gra[now].bs_varp)
                tmp = self.gra[now].value
                chlen = ptr.len/tmp
                for i in range(tmp):
                    chptr = BSVarType()
                    chptr.len = chlen
                    chptr.name = ptr.name + "[" + str(i) + "]"
                    chptr.value = 0
                    ptr.ch.push_back(chptr)
                beta = (1 << chlen) - 1
                for i in range(tmp-1, 0):
                    ptr.ch[i].value = ptr.value & beta
                    ptr.value = ptr.value >> chlen
                now += 1
            elif self.gra[now].type == 5:  # 对铭文进行合并
                ptr = self.find_bs_varp(self.gra[now].var, self.gra[now].bs_varp)
                tmp = ptr.ch.size()
                chlen = ptr.len/tmp
                ptr.value = 0
                for i in range(0, tmp):
                    chptr = ptr.ch[i]
                    ptr.value = (ptr.value << chlen) | chptr.value
                ptr.ch.clear()
                now += 1


if __name__ == "__main__":
    first_line = input("请按顺序输入n个置换表，m个S盒:")
    cipher = CIPHER(int(first_line.split(" ")[0]), int(first_line.split(" ")[1]))
    cipher.read_ptable()
    cipher.read_sbox()
    cipher.read_code()
    cipher.lex_ana()
    cipher.init_bs_var()
    cipher.gra_ana()
    cipher.solve_task1()


# 0 0
# state(10)
# key(1)
# BEGIN
# END
# 1
# 0000000000
# 1
#
# 1 1
# 4 4
# 1 2 3 0
# 4 4
# 0 1 2 3 3 2 1 0 12 13 14 15 15 14 13 12
# state(4)
# key(4)
# tmp(4)
# BEGIN
#     tmp=state
#     LOOP i 1 3
#         tmp = S[0](P[0](tmp)) + key
#     ENDLOOP
#     state = state +	tmp + "1111"
# END
# 1
# 0101
# 1100

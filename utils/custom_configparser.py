from typing import Dict, Iterator

from pyparsing import (Literal, CaselessLiteral, Word, Combine, Group, Optional,
                       ZeroOrMore, Forward, nums, alphas, oneOf, ParseException)
import math
import operator



import numpy as np
import configparser
from pprint import pprint




__source__ = '''http://pyparsing.wikispaces.com/file/view/fourFn.py
http://pyparsing.wikispaces.com/message/view/home/15549426
'''
__note__ = '''
All I've done is rewrap Paul McGuire's fourFn.py as a class, so I can use it
more easily in other places.
'''



class NumericStringParser(object):
    '''
    Most of this code comes from the fourFn.py pyparsing example

    '''

    def pushFirst(self, strg, loc, toks):
        self.exprStack.append(toks[0])

    def pushUMinus(self, strg, loc, toks):
        if toks and toks[0] == '-':
            self.exprStack.append('unary -')

    def __init__(self):
        """
        expop   :: '^'
        multop  :: '*' | '/'
        addop   :: '+' | '-'
        integer :: ['+' | '-'] '0'..'9'+
        atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
        factor  :: atom [ expop factor ]*
        term    :: factor [ multop factor ]*
        expr    :: term [ addop term ]*
        """
        point = Literal(".")
        e = CaselessLiteral("E")
        fnumber = Combine(Word("+-" + nums, nums) +
                          Optional(point + Optional(Word(nums))) +
                          Optional(e + Word("+-" + nums, nums)))
        ident = Word(alphas, alphas + nums + "_$")
        plus = Literal("+")
        minus = Literal("-")
        mult = Literal("*")
        div = Literal("/")
        lpar = Literal("(").suppress()
        rpar = Literal(")").suppress()
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")
        pi = CaselessLiteral("PI")
        expr = Forward()
        atom = ((Optional(oneOf("- +")) +
                 (ident + lpar + expr + rpar | pi | e | fnumber).setParseAction(self.pushFirst))
                | Optional(oneOf("- +")) + Group(lpar + expr + rpar)
                ).setParseAction(self.pushUMinus)
        # by defining exponentiation as "atom [ ^ factor ]..." instead of
        # "atom [ ^ atom ]...", we get right-to-left exponents, instead of left-to-right
        # that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor << atom + \
            ZeroOrMore((expop + factor).setParseAction(self.pushFirst))
        term = factor + \
            ZeroOrMore((multop + factor).setParseAction(self.pushFirst))
        expr << term + \
            ZeroOrMore((addop + term).setParseAction(self.pushFirst))
        # addop_term = ( addop + term ).setParseAction( self.pushFirst )
        # general_term = term + ZeroOrMore( addop_term ) | OneOrMore( addop_term)
        # expr <<  general_term
        self.bnf = expr
        # map operator symbols to corresponding arithmetic operations
        epsilon = 1e-12
        self.opn = {"+": operator.add,
                    "-": operator.sub,
                    "*": operator.mul,
                    "/": operator.truediv,
                    "^": operator.pow}
        self.fn = {"sin": math.sin,
                   "cos": math.cos,
                   "tan": math.tan,
                   "exp": math.exp,
                   "abs": abs,
                   "trunc": lambda a: int(a),
                   "round": round,
                   #"sgn": lambda a: abs(a) > epsilon and cmp(a, 0) or 0,
                   }

        self.exprStack = []


    def evaluateStack(self, s):
        op = s.pop()
        if op == 'unary -':
            return -self.evaluateStack(s)
        if op in "+-*/^":
            op2 = self.evaluateStack(s)
            op1 = self.evaluateStack(s)
            return self.opn[op](op1, op2)
        elif op == "PI":
            return math.pi  # 3.1415926535
        elif op == "E":
            return math.e  # 2.718281828
        elif op in self.fn:
            return self.fn[op](self.evaluateStack(s))
        elif op[0].isalpha():
            return 0
        else:
            return float(op)

    def eval(self, num_string, parseAll=True):
        try:
            self.exprStack = []
            results = self.bnf.parseString(num_string, parseAll)
            val = self.evaluateStack(self.exprStack[:])
            return val
        except ParseException:
            return num_string
        except AttributeError:
            return num_string








__numericStringParser = NumericStringParser()


def parse_list(l_str: str, is_numerical=True, dtype=None):
    l_str = l_str.replace("[","").replace("]","")
    l     = l_str.split(',')
    l     = map(lambda item: item.strip(), l)
    if is_numerical:
        return np.array([parse_expr(item, dtype) for item in l])
    else:
        return l

def parse_expr(expr_str, dtype=None):
    val = __numericStringParser.eval(expr_str, parseAll=True)
    if dtype is not None:
        return dtype(val)
    else:
        return val


def print_aligned(dict_: Dict):
    def to_aligned_records(dict_: Dict,
                           *,
                           sep: str = ' ') -> Iterator[str]:
        """Yields key-value pairs as strings that will be aligned when printed"""
        max_key_len = max(map(len, dict_.keys()))
        format_string = '{{key:{max_len}}} :{sep}{{value}}'.format(max_len=max_key_len, sep=sep)
        for key, value in dict_.items():
            yield format_string.format(key=key, value=value)

    print(*to_aligned_records(dict_), sep='\n')


class CustomConfigParser(configparser.ConfigParser):

    def get(self, section: str, option: str, **kwargs) -> str:
        try:
            return super().get(section, option, **kwargs)
        except configparser.NoOptionError:
            return None

    def getfloat(self, section: str, option: str, **kwargs) -> float:
        try:
            f_str = super().get(section, option, **kwargs)
            return parse_expr(f_str, dtype=float)
        except configparser.NoOptionError:
            return None

    def getint(self, section: str, option: str, **kwargs) -> int:
        try:
            i_str = super().get(section, option, **kwargs)
            return parse_expr(i_str, dtype=int)
        except configparser.NoOptionError:
            return None

    def getlist(self,section: str, option: str, is_numerical=True, dtype=None, **kwargs):
        try:
            l_str = super().get(section, option, **kwargs)
            return parse_list(l_str, is_numerical, dtype)
        except configparser.NoOptionError:
            return None


    def print(self, section=None, ignore_sections=('program_options','constants'), **kwargs):
        if ignore_sections is None:
            ignore_sections = set()

        if section is not None:
            print_aligned( self._sections[section] )
        else:
            sections = set(self.sections()) - set(ignore_sections)
            for section in sections:
                print("\n[{}]".format(section))
                print_aligned( self._sections[section] )
        print('')








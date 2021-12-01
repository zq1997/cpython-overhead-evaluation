from collections import namedtuple
import json
import subprocess

import numpy as np
from pygments.lexers import c_cpp

import config
from util import git
from util import saver

OPCODE_NUMBER = 256

OPCODE_CATEGORIES = [
    ('name access', '''
        LOAD_CONST
        LOAD_FAST   STORE_FAST  DELETE_FAST
        LOAD_DEREF  STORE_DEREF DELETE_DEREF
        LOAD_CLASSDEREF
        LOAD_GLOBAL STORE_GLOBAL    DELETE_GLOBAL
        LOAD_NAME   STORE_NAME  DELETE_NAME
    '''),
    ('attribute access', '''
        LOAD_ATTR   STORE_ATTR  DELETE_ATTR
        LOAD_METHOD
    '''),
    ('element access', '''
        BINARY_SUBSCR   STORE_SUBSCR    DELETE_SUBSCR
    '''),
    ('function call', '''
        CALL_FUNCTION
        CALL_FUNCTION_KW
        CALL_FUNCTION_EX
        CALL_METHOD
    '''),
    ('math operator', '''
        UNARY_INVERT
        UNARY_NEGATIVE
        UNARY_NOT
        UNARY_POSITIVE
        COMPARE_OP
        BINARY_ADD  INPLACE_ADD
        BINARY_AND  INPLACE_AND
        BINARY_FLOOR_DIVIDE INPLACE_FLOOR_DIVIDE
        BINARY_LSHIFT   INPLACE_LSHIFT
        BINARY_MATRIX_MULTIPLY  INPLACE_MATRIX_MULTIPLY
        BINARY_MODULO   INPLACE_MODULO
        BINARY_MULTIPLY INPLACE_MULTIPLY
        BINARY_OR   INPLACE_OR
        BINARY_POWER    INPLACE_POWER
        BINARY_RSHIFT   INPLACE_RSHIFT
        BINARY_SUBTRACT INPLACE_SUBTRACT
        BINARY_TRUE_DIVIDE  INPLACE_TRUE_DIVIDE
        BINARY_XOR  INPLACE_XOR
    '''),
    ('miscellany', '*')
]

Token = namedtuple('Token', ['text', 'line'])


class PoorAst:
    @staticmethod
    def create(source_content):
        ignore_kinds = {
            c_cpp.Text,
            *c_cpp.Comment.subtypes,
        }

        line = 1
        all_tokens = []
        for _, kind, text in c_cpp.CLexer().get_tokens_unprocessed(source_content):
            if kind not in ignore_kinds:
                all_tokens.append(Token(text, line))
            line += text.count('\n')
        all_tokens.append(Token(None, 0))
        all_tokens = iter(all_tokens)

        lefts = ('(', '[', '{')
        rights = (')', ']', '}')
        tree_types = ('()', '[]', '{}')

        def parse_tokens(end_with):
            tokens = []
            for token in all_tokens:
                token_text = token.text
                if token_text == end_with:
                    return tokens, token
                elif token_text in lefts:
                    i = lefts.index(token_text)
                    children, right_token = parse_tokens(rights[i])
                    token = PoorAst(children, tree_types[i], [token, right_token])
                tokens.append(token)
            else:
                assert False

        return PoorAst(parse_tokens(None)[0])

    def __init__(self, children, tree_type=None, extra=None):
        self.children = children
        self.tree_type = tree_type
        self.extra = extra

    def search(self, *texts, deep=False, backward=0, forward=1):
        def move_to_terminate(loc, step):
            while True:
                loc += step
                if not 0 <= loc < len(self.children):
                    break
                t = self.children[loc]
                if isinstance(t, PoorAst):
                    if t.tree_type == '{}':
                        break
                elif t.text == ';':
                    break
            return loc

        for i, t in enumerate(self.children):
            if isinstance(t, PoorAst):
                if deep and t.tree_type == '{}':
                    yield from t.search(*texts, deep=deep, backward=backward, forward=forward)
            elif t.text in texts:
                start_at = i
                end_at = i
                for _ in range(backward):
                    start_at = move_to_terminate(start_at, -1) + 1
                for _ in range(forward):
                    end_at = move_to_terminate(end_at, 1)
                yield self[start_at: end_at + 1]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.children[item]
        else:
            return PoorAst(self.children[item])

    def line_slice(self, internal=False):
        if self.tree_type is not None:
            left, right = self.extra
        else:
            left, right = self.children[0], self.children[-1]
        left_line = left.line_slice(True)[0] if isinstance(left, PoorAst) else left.line
        right_line = right.line_slice(True)[1] if isinstance(right, PoorAst) else right.line
        return (left_line, right_line) if internal else slice(left_line, right_line + 1)

    def block_entry_and_bracket_lines(self, entry_tok_count):
        left_line, right_line = self[:entry_tok_count].line_slice(True)
        lines = list(range(left_line, right_line + 1))
        if isinstance(self[-1], PoorAst) and self[-1].tree_type == '{}':
            left_bracket, right_bracket = self[-1].extra
            lines.append(left_bracket.line)
            lines.append(right_bracket.line)
        return lines


@saver.save_result_in_memory()
def read_ceval_c_file():
    text = git.read_file('Python/ceval.c')
    line_count = git.count_text_line(text) + 1
    func_ast, = PoorAst.create(text).search('_PyEval_EvalFrameDefault', backward=1)
    loop_ast, = func_ast[-1].search('main_loop')
    switch_ast, = loop_ast[-1].search('switch')
    case_asts = list(switch_ast[-1].search('TARGET', backward=1))
    return line_count, func_ast, loop_ast, switch_ast, case_asts


@saver.save_result_in_memory()
def get_opcode_names():
    proc = subprocess.Popen(
        [
            config.CPY_EXE,
            '-c', 'import opcode; import json; print(json.dumps(opcode.opname))'
        ],
        stdout=subprocess.PIPE
    )
    opcode_names = np.asarray(json.load(proc.stdout))
    assert proc.wait() == 0
    assert len(opcode_names) == OPCODE_NUMBER
    return opcode_names


def line_to_opcode():
    opcode_names = [
        *get_opcode_names(),
        '<NON_OPCODE>'
    ]
    line_count, func_ast, loop_ast, switch_ast, case_asts = read_ceval_c_file()
    remap_arr = np.full(line_count, OPCODE_NUMBER + 0, np.int)

    remap_arr[:] = OPCODE_NUMBER + 0  # <NON_OPCODE>
    func_slice = func_ast.line_slice()
    loop_slice = loop_ast.line_slice()
    switch_slice = switch_ast.line_slice()
    remap_arr[slice(func_slice.start, loop_slice.start)] = OPCODE_NUMBER + 1
    remap_arr[slice(loop_slice.start, switch_slice.start + 1)] = OPCODE_NUMBER + 2
    remap_arr[slice(switch_slice.stop, loop_slice.stop)] = OPCODE_NUMBER + 3
    remap_arr[slice(loop_slice.stop, func_slice.stop)] = OPCODE_NUMBER + 4

    for case_ast in case_asts:
        opcode = opcode_names.index(case_ast[2][0].text)
        remap_arr[case_ast.line_slice()] = opcode

    return remap_arr, np.asarray(opcode_names)


@saver.save_result_in_memory()
def line_to_opcode():
    opcode_names = get_opcode_names()
    line_count, _, _, _, case_asts = read_ceval_c_file()
    # N.B., -1 means invalid case
    remap_arr = np.full(line_count, -1, np.int)
    opcode_dict = {o: i for i, o in enumerate(opcode_names)}
    for case_ast in case_asts:
        remap_arr[case_ast.line_slice()] = opcode_dict[case_ast[2][0].text]
    return remap_arr, opcode_names


@saver.save_result_in_memory()
def eval_frame_setup_or_cleanup():
    line_count, func_ast, loop_ast, _, _ = read_ceval_c_file()
    remap_arr = np.full(line_count, False, np.bool)
    func_slice = func_ast.line_slice()
    remap_arr[func_slice] = True
    loop_slice = loop_ast.line_slice()
    remap_arr[loop_slice] = False
    return remap_arr


@saver.save_result_in_memory()
def line_to_stmt():
    stmt_names = ['non case', 'miscellany', 'dispatch', 'control flow', 'stack', 'ref count']
    idx_dispatch = 2
    idx_flow = 3
    line_count, func_ast, loop_ast, switch_ast, case_asts = read_ceval_c_file()
    remap_arr = np.zeros(line_count, np.int)

    remap_arr[next(loop_ast[-1].search('NEXTOPARG')).line_slice()] = idx_dispatch
    switch_entry_lines = switch_ast.block_entry_and_bracket_lines(2)
    remap_arr[switch_entry_lines] = idx_dispatch
    switch_entry = min(switch_entry_lines)
    for stmt in loop_ast[-1].search('if'):
        lines = stmt.block_entry_and_bracket_lines(2)
        if max(lines) >= switch_entry:
            break
        remap_arr[lines] = idx_dispatch

    for case_ast in case_asts:
        remap_arr[case_ast.line_slice()] = 1
        case_body_ast = case_ast[-1]

        for stmt in case_body_ast.search('DISPATCH', 'FAST_DISPATCH', deep=True):
            remap_arr[stmt.line_slice()] = idx_dispatch

        for stmt in case_body_ast.search('if', 'switch', 'while', 'for', deep=True):
            remap_arr[stmt.block_entry_and_bracket_lines(2)] = idx_flow
        for stmt in case_body_ast.search('else', 'do', deep=True):
            remap_arr[stmt.block_entry_and_bracket_lines(1)] = idx_flow
        for stmt in case_body_ast.search('break', 'continue', 'goto', deep=True):
            remap_arr[stmt.line_slice()] = idx_flow

        stack_macros = (
            'STACK_LEVEL', 'EMPTY',
            'TOP', 'SECOND', 'THIRD', 'FOURTH', 'PEEK',
            'SET_TOP', 'SET_SECOND', 'SET_THIRD', 'SET_FOURTH', 'SET_VALUE',
            'BASIC_STACKADJ', 'BASIC_PUSH', 'BASIC_POP',
            'PUSH', 'POP',
            'STACK_GROW', 'STACK_SHRINK',
            'EXT_POP'
        )
        for stmt in case_body_ast.search(*stack_macros, deep=True):
            if isinstance(stmt[-1], Token) and stmt[-1].text == ';':
                remap_arr[stmt.line_slice()] = 4

        for stmt in case_body_ast.search('Py_XINCREF', 'Py_INCREF', 'Py_XDECREF', 'Py_DECREF', deep=True):
            remap_arr[stmt.line_slice()] = 5

    return remap_arr, np.asarray(stmt_names)


@saver.save_result_in_memory()
def load_global_lines():
    # Hard-coded line numbers!
    line_count, *_ = read_ceval_c_file()
    remap_arr = np.zeros(line_count, np.int)
    remap_arr[2515:2600] = 4  # no opcache
    remap_arr[2515:2521] = 3  # common path
    remap_arr[2553:2570] = 2  # update opcache
    remap_arr[2521:2538] = 1  # test and use opcache
    return remap_arr


@saver.save_result_in_memory()
def opcode_to_category():
    remap_dict = {}
    category_names = []
    default_i = 0
    for i, (k, v) in enumerate(OPCODE_CATEGORIES):
        if v == '*':
            default_i = i
        category_names.append(k)
        for op in v.split():
            remap_dict[op] = i
    remap_arr = np.asarray([remap_dict.get(op, default_i) for op in get_opcode_names()])
    return remap_arr, np.asarray(category_names)


@saver.save_result_in_memory()
def is_dispatch_line():
    remap_arr, stmt_names = line_to_stmt()
    return remap_arr == list(stmt_names).index('dispatch')


@saver.save_result_in_memory()
def dispatch_for_opcodes(*opcodes):
    line_count, _, _, _, case_asts = read_ceval_c_file()
    remap_arr = np.zeros(line_count, np.bool)
    for case_ast in case_asts:
        if case_ast[2][0].text in opcodes:
            for stmt in case_ast[-1].search('DISPATCH', 'FAST_DISPATCH', deep=True):
                remap_arr[stmt.line_slice()] = True
    return remap_arr

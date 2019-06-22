#! /usr/bin/env python

BOOLEAN = ['True', 'False']


class Env(dict):

    def __init__(self, parms=(), args=(), outer=None):
        # Bind parm list to corresponding args, or single parm to list of args
        self.outer = outer
        if isinstance(parms, str):
            self.update({parms: list(args)})
        else:
            if len(args) != len(parms):
                raise TypeError('expected %s, given %s, '
                                % (to_string(parms), to_string(args)))
            self.update(zip(parms, args))

    def find(self, var):
        # "Find the innermost Env where var appears."
        if var in self:
            return self
        elif self.outer is None:
            raise LookupError(var)
        else:
            return self.outer.find(var)


def add_globals(env):
    import math, operator as op
    env.update(vars(math))
    env.update({
        '+': op.add, '-': op.sub, '*': op.mul, '/': op.floordiv, 'not': op.not_,
        '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le, '=': op.eq,
        'equal?': op.eq, 'eq?': op.is_
    })
    return env


global_env = add_globals(Env())


# ==========================================#

def tokenize(program):
    return [i for i in program.replace('(', ' ( ').replace(')', ' ) ').split(' ') if i]


def parse(program):
    # if len(program.split(" ")) == 1:
    #     return [program]
    return read_tokens(tokenize(program))


def read_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token)


def atom(token):
    try:
        return int(token)
    except Exception:
        try:
            return float(token)
        except Exception:
            return str(token)


def eval(x, env=global_env):
    try:
        if x in BOOLEAN:
            return x
        elif isinstance(x, str):  # variable reference
            return env.find(x)[x]
        elif not isinstance(x, list):  # constant literal
            return x
        elif x[0] == 'eq?':  # (if A equal B)
            (_, A, B) = x
            return "True" if eval(A, env) == eval(B, env) else "False"

        elif x[0] == 'cond':  #
            for i in x[1:]:
                if eval(i[0], env) == "True":
                    return eval(i[1], env)
        elif x[0] == 'define':  # (define var exp)
            (_, var, exp) = x
            env[var] = eval(exp, env)
            return "define"
        elif x[0] == 'lambda':  # (lambda (var*) exp)
            (_, vars, exp) = x
            return lambda *args: eval(exp, Env(vars, args, env))
        else:  # (proc exp*)
            exps = [eval(exp, env) for exp in x]
            proc = exps.pop(0)
            return proc(*exps)
    except Exception:
        print("semantic error")


def repl(prompt='--->'):
    while 1:
        program = input(prompt)
        if program:
            if program == '(quit)' or program == '(exit)':
                quit()
                break
            # if len(program.split(" ")) == 1:
            #     try:
            #         exec(program+"=0")
            #         print(">>STRING")
            #         continue
            #     except Exception:
            #         pass
            try:
                val = eval(parse(program))
            except Exception as e:
                print(parse(program))
                print("syntax error")
                continue
            if val is not None:
                print(">>", to_string(val))
                print(">>", "INT" if isinstance(val, int) else "Fun")


def to_string(exp):
    return '(' + ' '.join(map(to_string, exp)) + ')' if isinstance(exp, list) else str(exp)


def quit():
    print('Exit the interpreter...')


if __name__ == '__main__':
    repl()

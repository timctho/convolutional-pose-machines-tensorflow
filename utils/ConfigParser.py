class ConfigParser(object):
    def __init__(self, config_file=None):
        if config_file is not None:
            self.parser_config_file(config_file)

    def __getattr__(self, key):
        return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def _set_int(self, var_name, var_value):
        self.__setattr__(var_name, int(var_value))

    def _set_str(self, var_name, var_value):
        self.__setattr__(var_name, str(var_value))

    def _set_float(self, var_name, var_value):
        self.__setattr__(var_name, float(var_value))

    def _set_bool(self, var_name, var_value):
        self.__setattr__(var_name, bool(var_value))

    def parser_config_file(self, config_file):
        switch = {'i': self._set_int,
                  's': self._set_str,
                  'f': self._set_float,
                  'b': self._set_bool}

        with open(config_file, 'r') as config:
            for line in config:
                if len(line) == 1 or line.startswith('#'): continue # Skip empty line and comments
                tmp = line.strip().split('|')
                var_type, rest = tmp[0], tmp[1]
                var_list = rest.strip().split('=')
                var_name, var_value = var_list[0], var_list[1]
                var_name = var_name.strip()
                var_value = var_value.strip()

                try:
                    switch[var_type](var_name, var_value)
                except KeyError:
                    print(line.strip())
                    print('Variable type must be in i:int, f:float, s:string, b:bool\n')


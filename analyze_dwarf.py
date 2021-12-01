import json
import subprocess
import os

from elftools.elf.elffile import ELFFile

import config

PATH = config.CPY_EXE


def get_file_of_die(die):
    lp_header = die.cu.dwarfinfo.line_program_for_CU(die.cu).header
    file_index = die.attributes['DW_AT_decl_file'].value
    file_entry = lp_header['file_entry'][file_index - 1]
    dir_index = file_entry["dir_index"]
    dir_name = lp_header["include_directory"][dir_index - 1]
    file_name = file_entry.name
    return os.path.join(dir_name, file_name).decode()


def iter_typedef(type_die):
    yield type_die
    cu = type_die.cu
    cu_offset = cu.cu_offset
    while type_die.tag == 'DW_TAG_typedef':
        type_die = cu.get_DIE_from_refaddr(type_die.attributes['DW_AT_type'].value + cu_offset)
        yield type_die


def resolve_specification(die):
    if 'DW_AT_specification' not in die.attributes:
        return die
    cu = die.cu
    cu_offset = cu.cu_offset
    return cu.get_DIE_from_refaddr(die.attributes['DW_AT_specification'].value + cu_offset)


def get_types_of_die(die):
    cu = die.cu
    cu_offset = cu.cu_offset
    die = resolve_specification(die)
    if 'DW_AT_type' not in die.attributes:
        return
    type_die = cu.get_DIE_from_refaddr(die.attributes['DW_AT_type'].value + cu_offset)
    yield from iter_typedef(type_die)


def check_type(type_die, decl_file, decl_name):
    if type_die.tag != 'DW_TAG_structure_type' and type_die.tag != 'DW_TAG_typedef':
        return False
    if 'DW_AT_name' not in type_die.attributes:
        return False
    if type_die.attributes['DW_AT_name'].value.decode() != decl_name:
        return False
    if get_file_of_die(type_die) != decl_file:
        return False
    return True


def check_instance(type_die, decl_file, decl_name):
    for type_die in get_types_of_die(type_die):
        if check_type(type_die, decl_file, decl_name):
            return True
    return False


def iter_die(dwarf_info):
    for cu in dwarf_info.iter_CUs():
        top_die = cu.get_top_DIE()
        stack = [top_die]
        while stack:
            die = stack.pop()
            yield die
            stack.extend(die.iter_children())


def iter_die_of_type(elf_file, dwarf_info, struct_type, is_array):
    for die in iter_die(dwarf_info):
        if die.tag != 'DW_TAG_variable':
            continue
        attributes = die.attributes
        if 'DW_AT_location' not in attributes:
            continue

        array_die = die
        if is_array:
            types = tuple(get_types_of_die(die))
            if not types:
                continue
            array_die = types[-1]
            if array_die.tag != 'DW_TAG_array_type':
                continue

        if check_instance(array_die, struct_type.decl_file, struct_type.decl_name):
            var_file = get_file_of_die(die)
            die_sp = resolve_specification(die)
            if 'DW_AT_name' in die_sp.attributes:
                var_name = die_sp.attributes['DW_AT_name'].value.decode()
            else:
                var_name = '<unknown>'
            var_vaddr = int.from_bytes(die.attributes['DW_AT_location'].value[1:], 'little')

            total_size = struct_type.struct_size
            if is_array:
                for d in array_die.iter_children():
                    if d.tag == 'DW_TAG_subrange_type':
                        total_size *= d.attributes['DW_AT_upper_bound'].value + 1
                        break
                else:
                    continue
            var_bytes = read_elf_by_addr(elf_file, var_vaddr, total_size)
            if var_bytes is not None:
                yield var_file, var_name, var_vaddr, var_bytes


def read_elf_by_addr(elf_file, vaddr, size):
    file_offset = list(elf_file.address_offsets(vaddr))
    assert len(file_offset) <= 1
    if file_offset:
        file_offset = file_offset[0]
        stream = elf_file.stream
        stream.seek(file_offset)
        return stream.read(size)
    return b''


class StructType:
    def __init__(self, dwarf_info, decl_file, decl_name):
        for die in iter_die(dwarf_info):
            if check_type(die, decl_file, decl_name):
                die = tuple(iter_typedef(die))[-1]
                self.decl_name = decl_name
                self.decl_file = decl_file
                self.struct_size = die.attributes['DW_AT_byte_size'].value
                self.fields = {}
                for fd in die.iter_children():
                    if fd.tag == 'DW_TAG_member':
                        field_name = fd.attributes['DW_AT_name'].value.decode()
                        field_offset = fd.attributes['DW_AT_data_member_location'].value
                        field_size = tuple(get_types_of_die(fd))[-1].attributes['DW_AT_byte_size'].value
                        self.fields[field_name] = (field_offset, field_size)
                break
        else:
            raise Exception('cannot find struct %s in %s' % (decl_name, decl_file))

    def get_field(self, struct_bytes, field):
        field_offset, field_size = self.fields[field]
        field_bytes = struct_bytes[field_offset:field_offset + field_size]
        return int.from_bytes(field_bytes, 'little')


class Addr2line:
    def __init__(self, elf_path):
        self.proc = subprocess.Popen(
            ['addr2line', '-fe', elf_path],
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

    def __getitem__(self, addr):
        print('0x%x' % addr, file=self.proc.stdin, flush=True)
        func_name = self.proc.stdout.readline().rstrip()
        src_path, _ = self.proc.stdout.readline().rstrip().rsplit(':', 1)
        src_path = os.path.normpath(src_path)
        return src_path, func_name

    def close(self):
        self.proc.stdin.close()
        assert self.proc.wait() == 0


def find_native_func(elf_file, dwarf_info, addr2line):
    native_functions = set()

    type_struct = StructType(dwarf_info, '../../cpython/Include/methodobject.h', 'PyMethodDef')

    for result in iter_die_of_type(elf_file, dwarf_info, type_struct, True):
        var_file, var_name, var_vaddr, var_bytes = result
        ele_num = len(var_bytes) // type_struct.struct_size
        print('0x%x: struct PyMethodDef %s[%d] in %s' % (var_vaddr, var_name, ele_num, var_file))

        start = 0
        for i in range(ele_num):
            end = start + type_struct.struct_size
            func_addr = type_struct.get_field(var_bytes[start:end], 'ml_meth')
            start = end
            if func_addr:
                src_path, func_name = addr2line[func_addr]
                print('\t0x%x: %s() in %s' % (func_addr, func_name, src_path))
                native_functions.add((src_path, func_name))

    return native_functions

# Not useful
# def find_slot_func(elf_file, dwarf_info, addr2line):
#     type_struct = StructType(dwarf_info, '../../cpython/Include/object.h', 'PyTypeObject')
#     slots_struct = StructType(dwarf_info, '../../cpython/Include/cpython/object.h', 'PyNumberMethods')
#     number_slots = (
#         'nb_positive',
#         'nb_negative',
#         'nb_bool',
#         'nb_invert',
#
#         'nb_add',
#         'nb_subtract',
#         'nb_multiply',
#         'nb_floor_divide',
#         'nb_true_divide',
#         'nb_power',
#         'nb_remainder',
#         'nb_matrix_multiply',
#         'nb_and',
#         'nb_or',
#         'nb_xor',
#         'nb_lshift',
#         'nb_rshift',
#
#         'nb_inplace_add',
#         'nb_inplace_subtract',
#         'nb_inplace_multiply',
#         'nb_inplace_floor_divide',
#         'nb_inplace_true_divide',
#         'nb_inplace_power',
#         'nb_inplace_remainder',
#         'nb_inplace_matrix_multiply',
#         'nb_inplace_and',
#         'nb_inplace_or',
#         'nb_inplace_xor',
#         'nb_inplace_lshift',
#         'nb_inplace_rshift'
#     )
#
#     slot_functions = set()
#     for result in iter_die_of_type(elf_file, dwarf_info, type_struct, False):
#         var_file, var_name, var_vaddr, var_bytes = result
#         print('0x%x: struct PyTypeObject %s in %s' % (var_vaddr, var_name, var_file))
#
#         field_addr = type_struct.get_field(var_bytes, 'tp_as_number')
#         if field_addr:
#             field_bytes = read_elf_by_addr(elf_file, field_addr, slots_struct.struct_size)
#             for field in number_slots:
#                 func_addr = slots_struct.get_field(field_bytes, field)
#                 if func_addr:
#                     src_path, func_name = addr2line[func_addr]
#                     print('\t0x%x: [%s] %s() in %s' % (func_addr, field, func_name, src_path))
#                     slot_functions.add((src_path, func_name))
#
#         field_addr = type_struct.get_field(var_bytes, 'tp_richcompare')
#         if field_addr:
#             src_path, func_name = addr2line[field_addr]
#             print('\t0x%x: [richcompare] %s() in %s' % (field_addr, func_name, src_path))
#             slot_functions.add((src_path, func_name))
#
#     return slot_functions


def main():
    addr2line = Addr2line(PATH)

    with open(PATH, 'rb') as f:
        elf_file = ELFFile(f)
        if not elf_file.has_dwarf_info():
            raise RuntimeError(f'{PATH} has no DWARF info')
        dwarf_info = elf_file.get_dwarf_info()
        native_functions = find_native_func(elf_file, dwarf_info, addr2line)
    addr2line.close()

    with open(config.CPY_FUNC_JSON, 'wt') as f:
        data = sorted(native_functions)
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()

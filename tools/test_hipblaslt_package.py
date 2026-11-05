"""Configure-only dependency policy fixtures; fake libraries never link or run."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HipblasltPackage(unittest.TestCase):
    def setUp(self):
        self.cmake = shutil.which('cmake')
        self.ninja = shutil.which('ninja')
        if not self.cmake or not self.ninja:
            self.skipTest('CMake/Ninja required for compiler-free configure fixtures')
        scratch = ROOT / 'temp/hipblaslt-package-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def put(self, root, name, text='fixture; not a library'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def dependency(self, name='consumer SDK with spaces'):
        root = self.base / name
        self.put(root, 'include/hipblaslt/hipblaslt.h')
        library = self.put(root, 'lib/hipblaslt.lib' if os.name == 'nt' else 'lib/libhipblaslt.so')
        runtime = self.put(root, 'bin/hipblaslt.dll') if os.name == 'nt' else library
        self.put(root, 'lib/cmake/hipblaslt/hipblaslt-config.cmake',
            'add_library(roc::hipblaslt SHARED IMPORTED)\n'
            'add_library(hipblaslt_support INTERFACE IMPORTED)\n'
            'set_target_properties(roc::hipblaslt PROPERTIES\n'
            f' IMPORTED_LOCATION "{runtime.as_posix()}" IMPORTED_IMPLIB "{library.as_posix()}"\n'
            f' INTERFACE_INCLUDE_DIRECTORIES "{(root / "include").as_posix()}"\n'
            ' INTERFACE_LINK_LIBRARIES hipblaslt_support)\nset(hipblaslt_FOUND TRUE)\n')
        return root

    def package(self, enabled=True):
        stage = self.base / 'original RNS8 prefix'
        directory = stage / 'lib/cmake/RNS8'
        template = (ROOT / 'cmake/RNS8Config.cmake.in').read_text(encoding='utf-8')
        values = dict(RNS8_ENABLE_HIPBLASLT='ON' if enabled else 'OFF',
            RNS8_ENABLE_RCCL='OFF', RNS8_ENABLE_ROCSHMEM='OFF', RNS8_ENABLE_DISTRIBUTED_MPI='OFF',
            RNS8_ENABLE_HIP='OFF', CMAKE_INSTALL_BINDIR='bin', CMAKE_INSTALL_LIBDIR='lib',
            PACKAGE_INIT='get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)\n'
                         'macro(check_required_components name)\nendmacro()')
        self.put(stage, 'lib/cmake/RNS8/RNS8Config.cmake', re.sub(r'@([A-Za-z0-9_]+)@', lambda m: values.get(m[1], ''), template))
        for name in ('RNS8Runtime.cmake', 'RNS8HIPBLASLTDependency.cmake'):
            self.put(stage, 'lib/cmake/RNS8/' + name, (ROOT / 'cmake' / name).read_text(encoding='utf-8'))
        self.put(stage, 'lib/cmake/RNS8/RNS8Targets.cmake',
            'add_library(rns8::rns8_static INTERFACE IMPORTED)\n' +
            ('set_target_properties(rns8::rns8_static PROPERTIES INTERFACE_LINK_LIBRARIES roc::hipblaslt)\n' if enabled else ''))
        self.assertTrue(directory.is_dir())
        return stage

    def configure(self, stage, lines='', *, required=True, definitions=()):
        source = self.base / ('consumer-' + str(len(list(self.base.glob('consumer-*')))))
        self.put(source, 'CMakeLists.txt', 'cmake_minimum_required(VERSION 3.22)\nproject(DependencyPolicy NONE)\n'
            'set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)\nset(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY OFF)\n' +
            f'find_package(RNS8 CONFIG {"REQUIRED" if required else "QUIET"})\n' + lines)
        environment = {key: value for key, value in os.environ.items()
                       if key not in ('RNS8_HIP_ROOT', 'RNS8_HIPBLASLT_ROOT', 'HIP_PATH', 'ROCM_PATH')}
        return subprocess.run([self.cmake, '-S', str(source), '-B', str(source / 'build'), '-G', 'Ninja',
            '-DCMAKE_MAKE_PROGRAM=' + self.ninja, '-DCMAKE_PREFIX_PATH=' + str(stage), *definitions],
            capture_output=True, text=True, encoding='utf-8', env=environment, timeout=40)

    def test_nondefault_consumer_sdk_preserves_complete_upstream_target_after_package_move(self):
        sdk = self.dependency()
        stage = self.package()
        moved = self.base / 'moved RNS8 prefix'
        stage.rename(moved)
        result = self.configure(moved,
            'if(NOT TARGET rns8::rns8_static OR NOT TARGET roc::hipblaslt)\nmessage(FATAL_ERROR "missing imported targets")\nendif()\n'
            'get_target_property(links roc::hipblaslt INTERFACE_LINK_LIBRARIES)\n'
            'if(NOT links STREQUAL "hipblaslt_support")\nmessage(FATAL_ERROR "lost upstream link interface")\nendif()\n',
            definitions=['-DRNS8_HIPBLASLT_ROOT=' + str(sdk)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_config_with_raw_files_is_an_explicit_required_or_quiet_rejection(self):
        sdk = self.dependency()
        (sdk / 'lib/cmake/hipblaslt/hipblaslt-config.cmake').unlink()
        stage = self.package()
        definitions = ['-DRNS8_HIPBLASLT_ROOT=' + str(sdk)]
        result = self.configure(stage, definitions=definitions)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('raw-library fallback is unsupported', ' '.join((result.stdout + result.stderr).split()))
        result = self.configure(stage, 'if(RNS8_FOUND OR TARGET rns8::rns8_static)\nmessage(FATAL_ERROR "false package success")\nendif()\n',
                                required=False, definitions=definitions)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_or_conflicting_explicit_consumer_roots_do_not_fall_back(self):
        stage = self.package()
        one = self.dependency('selected SDK')
        two = self.dependency('different SDK')
        for definitions in (['-DRNS8_HIPBLASLT_ROOT=' + str(self.base / 'missing')],
                            ['-DRNS8_HIPBLASLT_ROOT=' + str(one), '-Dhipblaslt_DIR=' + str(two / 'lib/cmake/hipblaslt')]):
            result = self.configure(stage, definitions=definitions)
            self.assertNotEqual(result.returncode, 0)

    def test_cpu_only_package_does_not_require_optional_sdk(self):
        stage = self.package(enabled=False)
        result = self.configure(stage, 'if(NOT TARGET rns8::rns8_static)\nmessage(FATAL_ERROR "missing CPU target")\nendif()\n',
                                definitions=['-DRNS8_HIPBLASLT_ROOT=' + str(self.base / 'missing')])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_partial_upstream_import_cannot_become_success_on_a_second_quiet_lookup(self):
        sdk = self.dependency()
        package = sdk / 'lib/cmake/hipblaslt/hipblaslt-config.cmake'
        package.write_text(package.read_text(encoding='utf-8') +
            '\nset(hipblaslt_FOUND FALSE)\nset(hipblaslt_NOT_FOUND_MESSAGE "missing transitive dependency")\n', encoding='utf-8')
        result = self.configure(self.package(),
            'find_package(RNS8 CONFIG QUIET)\n'
            'if(RNS8_FOUND OR TARGET rns8::rns8_static)\nmessage(FATAL_ERROR "accepted partial dependency")\nendif()\n',
            required=False, definitions=['-DRNS8_HIPBLASLT_ROOT=' + str(sdk)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

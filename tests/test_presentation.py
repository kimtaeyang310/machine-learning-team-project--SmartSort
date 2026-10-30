"""Presentation settings must travel with the repo and ignore the current cwd."""
# 작업 위치가 바뀌어도 시연 자료를 찾고 잘못된 설정 경로를 거부하는지 검사합니다.
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.paths import ROOT,project_path
from src.presentation import config_path,load_presentation
import main as entry


class PortablePresentationTests(unittest.TestCase):
    def test_project_paths_do_not_depend_on_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            old=Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(project_path('models/presentation'),(ROOT/'models/presentation').resolve())
            finally:os.chdir(old)

    def test_config_rejects_machine_paths_and_parent_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            self.assertEqual(config_path('data/photo.jpg',root),root/'data/photo.jpg')
            for bad in ('C:/Users/someone/model.pt',str(root/'absolute.png'),'../outside.png'):
                with self.assertRaises(ValueError):config_path(bad,root)

    def test_config_resolves_inside_a_relocated_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('data/photo.png','models/demo/model.pt','models/demo/policy.joblib','models/demo/split_manifest.json','models/demo/bundle.json'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test')
            config=root/'config/demo.json';config.parent.mkdir()
            config.write_text(json.dumps({'images':['data/photo.png'],'model_dir':'models/demo','source':'outputs/input.mp4','output':'outputs/result.mp4'}),encoding='utf-8')
            _,images,model,source,output=load_presentation('config/demo.json',root=root)
            self.assertEqual(model,(root/'models/demo').resolve())
            self.assertEqual(images,[(root/'data/photo.png').resolve()])
            self.assertFalse(source.exists())

    def test_menu_has_only_one_video_choice_and_dispatches_it(self):
        with patch('builtins.input',return_value='6'),patch('builtins.print') as output,patch('src.presentation.run_presentation') as demo:
            entry.main()
        demo.assert_called_once_with(display=True)
        lines=[str(call.args[0]) for call in output.call_args_list if call.args]
        self.assertIn('6. 선별 테스트 과일 시연',lines)
        self.assertFalse(any(line.startswith(('7.','8.')) for line in lines))


if __name__=='__main__':unittest.main()

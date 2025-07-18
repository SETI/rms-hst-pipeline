import os
import pytest
from unittest import mock
import HST.hst_helper.general_utils as general_utils

@pytest.fixture
def fake_logger():
    class FakeLogger:
        def __init__(self):
            self.messages = []
        def info(self, msg):
            self.messages.append(msg)
        def error(self, msg, *args):
            self.messages.append(f'ERROR: {msg}')
    return FakeLogger()

def test_create_collection_label_creates_label(monkeypatch, fake_logger):
    called = {}
    def fake_create_xml_label(template_path, label_path, data_dict, logger):
        called['template_path'] = template_path
        called['label_path'] = label_path
        called['data_dict'] = data_dict
    monkeypatch.setattr(general_utils, 'create_xml_label', fake_create_xml_label)
    monkeypatch.setattr(general_utils, 'get_deliverable_path', lambda pid, testing: '/fake/deliverable')
    result = general_utils.create_collection_label('12345', 'data', {'foo': 'bar'}, 'label.xml', 'template.xml', fake_logger, testing=True)
    assert result == '/fake/deliverable/data/label.xml'
    assert called['template_path'].endswith('template.xml')
    assert called['data_dict'] == {'foo': 'bar'}

def test_create_xml_label_calls_template(monkeypatch, fake_logger):
    class FakeTemplate:
        ERROR_COUNT = 0
        def __init__(self, path):
            self.path = path
        def write(self, data_dict, label_path):
            self.data_dict = data_dict
            self.label_path = label_path
        @staticmethod
        def set_logger(logger):
            pass
    monkeypatch.setattr(general_utils, 'PdsTemplate', FakeTemplate)
    general_utils.create_xml_label('template', 'label', {'foo': 'bar'}, fake_logger)
    # No assertion needed, just check no error

def test_create_xml_label_error_count(monkeypatch, fake_logger):
    class FakeTemplate:
        def __init__(self, path):
            self.path = path
        def write(self, data_dict, label_path):
            pass
        @staticmethod
        def set_logger(logger):
            pass
    monkeypatch.setattr(general_utils, 'PdsTemplate', FakeTemplate)
    # Test error count 1
    FakeTemplate.ERROR_COUNT = 1
    general_utils.create_xml_label('template', 'label', {'foo': 'bar'}, fake_logger)
    assert any('ERROR:' in m for m in fake_logger.messages)
    # Test error count > 1
    FakeTemplate.ERROR_COUNT = 2
    fake_logger.messages.clear()
    general_utils.create_xml_label('template', 'label', {'foo': 'bar'}, fake_logger)
    assert any('ERROR:' in m for m in fake_logger.messages)

def test_create_csv(tmp_path, fake_logger):
    csv_path = tmp_path / 'test.csv'
    data = [[1, 2, 3], [4, 5, 6]]
    general_utils.create_csv(str(csv_path), data, fake_logger)
    with open(csv_path) as f:
        lines = f.read().splitlines()
    assert lines == ['1,2,3', '4,5,6']

def test_get_citation_info_from_dict(monkeypatch):
    monkeypatch.setitem(general_utils.CITATION_INFO_DICT, '12345', 'info')
    assert general_utils.get_citation_info('12345', None) == 'info'

def test_get_citation_info_from_file(monkeypatch, tmp_path, fake_logger):
    general_utils.CITATION_INFO_DICT.pop('54321', None)  # Ensure key is not present
    monkeypatch.setattr(general_utils, 'get_formatted_proposal_id', lambda pid: '54321')
    monkeypatch.setattr(general_utils, 'get_program_dir_path', lambda pid, v, root_dir: str(tmp_path))
    monkeypatch.setattr(general_utils, 'DOCUMENT_EXT_FOR_CITATION_INFO', ['txt'])
    class FakeCitation:
        @staticmethod
        def create_from_file(path):
            return 'created'
    monkeypatch.setattr(general_utils, 'Citation_Information', FakeCitation)
    file_path = tmp_path / 'file.txt'
    file_path.write_text('dummy')
    result = general_utils.get_citation_info('54321', fake_logger)
    assert result == 'created'

def test_get_instrument_id_set(monkeypatch, fake_logger):
    monkeypatch.setitem(general_utils.INST_ID_DICT, '12345', set(['WFC']))
    monkeypatch.setattr(general_utils, 'get_formatted_proposal_id', lambda pid: '12345')
    assert general_utils.get_instrument_id_set('12345', fake_logger) == set(['WFC'])

def test_get_mod_history_from_label(monkeypatch, tmp_path):
    xml = '<xml></xml>'
    file_path = tmp_path / 'label.xml'
    file_path.write_text(xml)
    monkeypatch.setattr(os.path, 'exists', lambda p: True)
    monkeypatch.setattr('builtins.open', mock.mock_open(read_data=xml))
    monkeypatch.setattr(general_utils, 'get_modification_history', lambda xml: [("2020-01-01", "1.0", "desc")])
    result = general_utils.get_mod_history_from_label(str(file_path), '2.0')
    assert result == [("2020-01-01", "1.0", "desc")]

def test_date_time_to_date():
    assert general_utils.date_time_to_date('2005-01-19T15:41:05Z') == '2005-01-19'
    with pytest.raises(ValueError):
        general_utils.date_time_to_date('2005-01-19')

def test_is_browse_prod(monkeypatch):
    monkeypatch.setattr(general_utils, 'BROWSE_PROD_EXT', ['.jpg', '.png'])
    assert general_utils.is_browse_prod('foo.jpg')
    assert not general_utils.is_browse_prod('foo.fits')

def test_get_target_id_from_label_new(monkeypatch, tmp_path):
    # Setup: TARG_ID_DICT does not have the key, file exists, xml_content is not empty
    xml_content = '<xml></xml>'
    file_path = tmp_path / 'label.xml'
    file_path.write_text(xml_content)
    monkeypatch.setattr(os.path, 'exists', lambda p: True)
    monkeypatch.setattr('builtins.open', mock.mock_open(read_data=xml_content))
    monkeypatch.setattr(general_utils, 'get_formatted_proposal_id', lambda pid: '99999')
    monkeypatch.setattr(general_utils, 'get_target_identifications', lambda xml: [('name', ['alt'], 'type', ['desc'], 'lid')])
    if '99999' in general_utils.TARG_ID_DICT:
        del general_utils.TARG_ID_DICT['99999']
    result = general_utils.get_target_id_from_label('99999', str(file_path))
    assert result == [('name', ['alt'], 'type', ['desc'], 'lid')]

def test_get_target_id_from_label_existing(monkeypatch):
    # Setup: TARG_ID_DICT already has the key
    general_utils.TARG_ID_DICT['88888'] = [('foo', [], '', [], '')]
    monkeypatch.setattr(general_utils, 'get_formatted_proposal_id', lambda pid: '88888')
    result = general_utils.get_target_id_from_label('88888', '/does/not/matter.xml')
    assert result == [('foo', [], '', [], '')]

def test_get_collection_label_data(monkeypatch, tmp_path, fake_logger):
    # Setup: minimal directory with one xml file, all DICTs empty
    proposal_id = '77777'
    collection_name = 'col'
    formatted_id = '77777'
    if formatted_id in general_utils.TARG_ID_DICT:
        del general_utils.TARG_ID_DICT[formatted_id]
    general_utils.TIME_DICT[formatted_id] = {}
    general_utils.INST_PARAMS_DICT[formatted_id] = {}
    general_utils.PRIMARY_RES_DICT[formatted_id] = {}
    general_utils.RECORDS_DICT[formatted_id] = {}
    xml_content = '<xml></xml>'
    xml_file = tmp_path / 'file.xml'
    xml_file.write_text(xml_content)
    monkeypatch.setattr(general_utils, 'get_formatted_proposal_id', lambda pid: formatted_id)
    monkeypatch.setattr(general_utils, 'get_format_term', lambda fname: 'file')
    monkeypatch.setattr(general_utils, 'is_browse_prod', lambda fname: False)
    monkeypatch.setattr(general_utils, 'get_target_identifications', lambda xml: [('n', [], '', [], '')])
    monkeypatch.setattr(general_utils, 'get_time_coordinates', lambda xml: ('start', 'stop'))
    monkeypatch.setattr(general_utils, 'get_instrument_params', lambda xml: 'params')
    monkeypatch.setattr(general_utils, 'get_primary_result_summary', lambda xml: 'summary')
    res = general_utils.get_collection_label_data(proposal_id, str(tmp_path), fake_logger)
    assert res['target'] == [('n', [], '', [], '')]
    assert res['time'] == ('start', 'stop')
    assert res['inst_params'] == 'params'
    assert res['primary_res'] == 'summary'
    assert res['records'] == 1

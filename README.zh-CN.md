# UniARC

���ֿ��������� **Discrete vs. Continuous: A Comprehensive Study of Unified Audio
Understanding in LALMs** ������ʵ��·�ߣ��Ƚ��������������������е�������������ɢ������

**���״����Ѿ�������һ���ֿ���**��ԭ `UniARC/code` ������ `uniarc/`��
ԭ `UniARC_paper/xares-llm` ������ `xares-llm/`��ֻ���ȡ���ֿ⣬�������Ӧ·��ʹ�ã�
�����ٷֱ���������ԭ�ֿ⡣

[English](README.md) �� [����������Ӧ](docs/paper_mapping.md) ��
[���ݺ�ģ��׼��](docs/data_and_models.md) �� [�ϲ���¼](docs/migration.md)

| ʵ��·�� | ����Ŀ¼ | ����ģ�� | ѵ����ʽ |
| --- | --- | --- | --- |
| ������Ч΢�� | `xares-llm/` | SmolLM2 135M / 360M | ͶӰ���� LoRA��20 ������9 ������� |
| ����Ǹ�̽�� | `uniarc/` | Llama 1B / 8B | �������ԹǸɣ�ѵ������㣬7 �����񸲸� 8 �����ݼ� |

������ **UniARC** ��Ϊͳһ������ƣ�XARES-LLM ��΢��·��ʹ�õ����ο�ܡ�
Դ���б����� WavTokenizer �Ķ���̽����չ�������ı� 2 û�б�����һ��������

���汾����Դ�롢��������ڡ�˵���ĵ�����Դ��¼���鹤�ߣ����ݼ���Ԥѵ��Ȩ�غ�
��ѵ���������Ҫ����׼��������ʵ�������������Ĳ�������Ķ�Ӧ�ĵ������ܰ�Դ����
��С��ģ������Ϊ���Ľ���Ѿ����֡�

## ���ٿ�ʼ

```bash
git clone https://github.com/Anachronism-N/UniARC.git
cd UniARC
```

������ Linux + NVIDIA GPU ����ѵ��������·��ʹ�ö������⻷������װ����ֱ��
[����̽��˵��](uniarc/README.md) �� [XARES-LLM ˵��](xares-llm/README.md)��
��Ҫ����������ǿ�а�װ��ͬһ������

�ڲֿ��Ŀ¼������ʹ�� Python 3.10+ ִ�в���Ҫ����ģ�͵ļ�飺

```bash
python run.py --help
python scripts/check_release.py
python -m unittest discover -s tests -v
python run.py probe --help
python run.py --show-command xares example/hubert-large/hubertlarge.py cremad cremad
```

`--show-command` ֻչʾҪִ�е������빤��Ŀ¼��

### ʹ�� UniARC ����̽�����

�ڲֿ��Ŀ¼������������������Ϊ Bash �����

```bash
python3.11 -m venv .venv-probe
source .venv-probe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/probe.txt
```

�� [uniarc/README.md](uniarc/README.md) ׼�����ݺ�ģ�ͣ����޸� JSON �����е�λ�á�
������ü�ѵ����

```bash
python run.py probe train --config uniarc/configs/hubert_asr.json --dry-run
python run.py probe train --config uniarc/configs/hubert_asr.json
```

����ǰ���� `hubert_asr_infer.json` ����дѵ���õ��� `checkpoint` �� `data.test`��

```bash
python run.py probe infer --config uniarc/configs/hubert_asr_infer.json
```

�������Ϊ���õ����Ŀ¼�µ� `predictions.jsonl`������������
[����˵��](docs/evaluation.md)��

### ʹ�� XARES-LLM ΢������

����һ���նˣ���ͬһ�ֿ��Ŀ¼���� XARES ������Bash����

```bash
python3.11 -m venv .venv-xares
source .venv-xares/bin/activate
python -m pip install --upgrade pip
python -m pip install -c xares-llm/requirements-constraints.txt -e ./xares-llm
```

��Ҫ�� CUDA ƥ��� PyTorch���Լ��� TorchCodec ʹ�õ� FFmpeg�����尲װ��ģ����Դ��
[xares-llm/README.md](xares-llm/README.md)����������ѵ�������� CREMA-D��

```bash
python run.py xares example/hubert-large/hubertlarge.py cremad cremad \
  --args '{"decoder_model_name":"HuggingFaceTB/SmolLM2-135M","projector_type":"mlp","lora_enabled":true}'
```

��ģ������Ϊ `HuggingFaceTB/SmolLM2-360M` ���л�����һ��ģ����ɢ����������Ҫ׼��
��Ӧ�뱾��Ȩ�أ�������˵���������������һ��ʵ�飬�����Զ������ƪ���ĵ�ȫ��ʵ�顣

ѵ����ɺ��Զ����������д��
`xares-llm/experiments/<config>/<decoder>/<encoder>/scores.tsv`��

ͳһ��ڻ��� `xares-llm/` Ŀ¼������΢����ˣ�����������Զ��� YAML ·������ڸ�Ŀ¼��
�����˵Ĺ���Ŀ¼�ǲֿ��Ŀ¼��`--python` ��ָ����һ�����⻷���� Python ��ִ���ļ���

## ������������

- ͳһ�������ơ�˫·�߽ṹ��������ں�����׼��˵����
- ���о��ű��еĻ���·����Ϊ��������Դ���޸�Ӱ��Ǩ�ƵĴ������⡣
- ������Ҫ�������������ã��ų���־�����桢�������Git ���ݺ��޹�ʵ�顣
- ���� Apache-2.0���������ΰ�Ȩ���ļ���Դ���ṩ����������Ϣ��
- ���Ӽ��ű��ͳ����������ã�ʵ��ִ�з�Χ��[��֤��¼](docs/validation.md)��

## ����������֤

��������������Ϊ�����ύ����� `CITATION.cff` ��ʱ����¼�������á�
��ʽ��������˳��DOI �ͷ�����Ϣȷ�Ϻ�����д���ɲֿ��еľɱ��������δֱ�����á�

��Ŀ������� [Apache-2.0](LICENSE)�������������� [NOTICE](NOTICE)��������ģ��Ȩ��
��ѭ�������ɡ����ֿ�ֻ�������뼰���ã�������ѵ�����ݻ�ģ��Ȩ�ء�

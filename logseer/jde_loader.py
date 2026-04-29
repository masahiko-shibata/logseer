import re
from .loader import Loader


# JDE kernel process types detected from log file headers
_JDE_KERNEL_TYPES = [
    'UBE', 'SECURITY', 'WORK FLOW', 'CALLOBJ', 'METADATA',
    'MANAGEMENT', 'TEXTSEARCH', 'SAW', 'PACKAGE', 'QUEUE', 'XML',
]
_JDE_KERNEL_TYPE_TAGS = [
    'UBE', 'SECURITY', 'WORKFLOW', 'CALLOBJ', 'METADATA',
    'MANAGEMENT', 'TEXTSEARCH', 'SAW', 'PACKAGE', 'QUEUE', 'XML',
]


class JDELoader(Loader):
    """Loader for JD Edwards EnterpriseOne server logs.

    Extends the generic Loader with:
    - JDE kernel process type tagging (UBE, CALLOBJ, JDENET, etc.)
    - JDE-specific log cleaning (message IDs, kernel references, network errors)
    """

    def extract_file_tag(self, file):
        """Detect JDE kernel type or JDENET from log file header and return a tag."""
        index = file.find('Kernel of Type:', 0, 200)
        if index != -1:
            for ktype, tag in zip(_JDE_KERNEL_TYPES, _JDE_KERNEL_TYPE_TAGS):
                if file.find(ktype, index + 16, index + 26) != -1:
                    return tag
            return 'KERNEL'
        if file.find('jdenet_n> registered', 0, 200) != -1:
            return 'JDENET'
        return ''

    def gen_labeled_texts(self, file_groups, label_id, numchar=3000, multiple=1):
        """Extend base gen_text_label to print JDE-specific log type counts."""
        num_k = sum(
            1 for item in file_groups
            for file in item[0]
            if file.find('Kernel of Type:', 0, 200) != -1
        )
        num_n = sum(
            1 for item in file_groups
            for file in item[0]
            if file.find('jdenet_n> registered', 0, 200) != -1
        )
        result = super().gen_labeled_texts(file_groups, label_id, numchar=numchar, multiple=multiple)
        print('Found %s kernel logs.' % num_k)
        print('Found %s jdenet logs.' % num_n)
        return result

    def clean_domain(self, t):
        """JDE-specific normalization: message IDs, kernel refs, network errors, paths."""
        # Directory paths
        t = re.sub(r'/slot/[A-Za-z0-9/]+/', '', t)
        # JDE message and kernel IDs
        t = re.sub(r'(^|\s)msgId[=\-][0-9]+[\.]*(\s|$)', 'msgId=num ', t)
        t = re.sub(r'(^|\s)msgPort[=\-][0-9]+[\.]*(\s|$)', 'msgPort=num ', t)
        t = re.sub(r'(^|\s)reqKrnl[=\-][0-9]+[\.]*(\s|$)', 'reqKrnl=num ', t)
        t = re.sub(r'(^|\s)reqKrnl[=\-][0-9]+[\.]*(\s|$)', 'resKrnl=num ', t)
        t = re.sub(r'(^|\s)reqNet[=\-][0-9]+[\.]*(\s|$)', 'reqNet=num ', t)
        t = re.sub(r'(^|\s)resNet[=\-][0-9]+[\.]*(\s|$)', 'resNet=num ', t)
        t = re.sub(r'(^|\s)maxrows=[0-9]+(\s|$)', ')maxrows=num  ', t)
        t = re.sub(r'(^|\s)fetched=[0-9]+(\s|$)', 'fetched=num  ', t)
        t = re.sub(r'<Krnl[0-9]+ReqQ>', '<Krnl_num_ReqQ>', t)
        t = re.sub(r'(^|\s)conn=[0-9a-f]{8}(\s|$)', 'conn=hex  ', t)
        t = re.sub(r'(^|\s)requ=[0-9a-f]{8}(\s|$)', 'requ=hex  ', t)
        # Network errors
        t = re.sub(r'le_net_error [0-9][0-9]:<.+?> <.*?>', 'le_net_error num:', t)
        return t


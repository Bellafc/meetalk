def convert_to_tree(result_list):
    root = Node("Part Chapters")
    chapter_dict = {}
    for item in result_list:
        chapter_name = "Chapter" + item["chapter"]
        section_name = "Section" + item["section"]
        if chapter_name not in chapter_dict:
            chapter_node = Node(chapter_name)
            chapter_dict[chapter_name] = chapter_node
            root.add_child(chapter_node)
        else:
            chapter_node = chapter_dict[chapter_name]
        section_node = Node(section_name)
        chapter_node.add_child(section_node)
    return root


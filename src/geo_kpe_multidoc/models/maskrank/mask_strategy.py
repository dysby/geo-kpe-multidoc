import re
from typing import Protocol

from loguru import logger

from geo_kpe_multidoc.document import Document


class CandidateMaskEmbeddingStrategy(Protocol):
    def candidate_embeddings(self, model, doc: Document):
        ...


class MaskFirst:
    occurrences = 1

    def candidate_embeddings(
        self, model, doc: Document, mask_token: str, text_prefix: str = ""
    ):
        # if cand_mode == "MaskFirst" or cand_mode == "MaskAll":
        #     occurences = 1 if cand_mode == "MaskFirst" else 0
        # TODO: does not work (candidate is not found)
        escaped_docs = []

        for _, mentions in doc.candidate_mentions.items():
            text = doc.raw_text
            # TODO: validation candidate is in text
            found_candidate = False
            for mention in sorted(mentions, key=len, reverse=True):
                if mention in text:
                    found_candidate = True
                    text = text.replace(mention, mask_token, self.occurrences)
                    if self.occurrences == 1:
                        break

            if not found_candidate:
                logger.debug(f"Mentions '{mentions}' - not in text")
            escaped_docs.append(text_prefix + text)

        embd = model.encode_batch(escaped_docs)
        doc.candidate_set_embed = [embd[i] for i in range(len(embd))]


class MaskAll(MaskFirst):
    occurrences = -1

    # escaped_docs = [
    #     re.sub(re.escape(candidate), "<mask>", doc.raw_text, occurences)
    #     for candidate in doc.candidate_set
    # ]
    # doc.candidate_set_embed = self.model.embed(escaped_docs)


class MaskHighest:
    def candidate_embeddings(
        self, model, doc: Document, mask_token: str, text_prefix: str = ""
    ):
        for candidate in doc.candidate_set:
            candidate = re.escape(candidate)
            candidate_embeds = []

            # TODO: what if not found?
            # TODO: refactor to batch processing
            for match in re.finditer(candidate, doc.raw_text):
                masked_text = f"{doc.raw_text[:match.span()[0]]}{mask_token}{doc.raw_text[match.span()[1]:]}"
                candidate_embeds.append(model.encode(text_prefix + masked_text))
                # if attention == "global_attention":
                #     candidate_embeds.append(self._embed_global(masked_text))
                # else:
                #     candidate_embeds.append(self.model.embed(masked_text))

        doc.candidate_set_embed.append(candidate_embeds)


class MaskSubset:
    def candidate_embeddings(
        self, model, doc: Document, mask_token: str, text_prefix: str = ""
    ):
        doc.candidate_set = sorted(
            doc.candidate_set, reverse=True, key=lambda x: len(x)
        )
        seen_candidates = {}

        for candidate in doc.candidate_set:
            prohibited_pos = []
            len_candidate = len(candidate)
            for prev_candidate in seen_candidates:
                if len_candidate == len(prev_candidate):
                    break

                elif candidate in prev_candidate:
                    prohibited_pos.extend(seen_candidates[prev_candidate])

            pos = [
                (match.span()[0], match.span()[1])
                for match in re.finditer(re.escape(candidate), doc.raw_text)
            ]

            seen_candidates[candidate] = pos
            subset_pos = []
            for p in pos:
                subset_flag = True
                for prob in prohibited_pos:
                    if p[0] >= prob[0] and p[1] <= prob[1]:
                        subset_flag = False
                        break
                if subset_flag:
                    subset_pos.append(p)

            masked_doc = doc.raw_text
            for i in range(len(subset_pos)):
                masked_doc = f"{masked_doc[:(subset_pos[i][0] + i*(len_candidate - 5))]}{mask_token}{masked_doc[subset_pos[i][1] + i*(len_candidate - 5):]}"

            doc.candidate_set_embed.append(model.embed(text_prefix + masked_doc))
